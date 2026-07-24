#!/usr/bin/env python3
"""
validate_video_vs_imu_v3.py
===========================

Kompakte, repository-spezifische Orchestrierung für die externe Validierung
von Lucas–Kanade-Video-Tracking gegen archivierte IMU-Messungen.

Version 3 verwendet die bewährte Einzelvalidierung aus
``validate_video_vs_imu_v2.py`` und ergänzt:

* robuste IMU-Zeitaufbereitung aus ``day`` + ``time``;
* automatischen Vergleich von linx, liny, linz, rotx, roty und rotz;
* eine gemeinsame ``comparison.csv``;
* ``summary.json`` und ``summary.html``;
* Ranking nach Korrelation und Frequenzübereinstimmung;
* gemeinsame Übersichtsplots;
* Hinweise zur physikalischen Vergleichsgröße und Vorverarbeitung.

Die Datei ersetzt nicht die Fachmodule im Paket ``validation``. Sie hält die
Einzelpipeline in v2 unverändert wiederverwendbar und vermeidet dadurch eine
zweite, lange Kopie derselben Loader-, Synchronisations-, Metrik- und
Plotting-Logik.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import logging
import math
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Ausführung über ``python src/validate_video_vs_imu_v3.py`` unterstützen.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from validate_video_vs_imu_v2 import (  # noqa: E402
    ValidationConfig,
    ValidationResult,
    validate_video_vs_imu,
)

LOGGER = logging.getLogger("validate_video_vs_imu_v3")
DEFAULT_AXES = ("linx", "liny", "linz", "rotx", "roty", "rotz")


@dataclass(slots=True)
class AxisResult:
    """Verdichtetes Ergebnis eines einzelnen Achsenlaufs."""

    axis: str
    status: str
    output_dir: str
    pearson_correlation: Optional[float] = None
    absolute_correlation: Optional[float] = None
    correlation_at_lag: Optional[float] = None
    lag_s: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r_squared: Optional[float] = None
    samples: Optional[int] = None
    duration_s: Optional[float] = None
    video_dominant_frequency_hz: Optional[float] = None
    imu_dominant_frequency_hz: Optional[float] = None
    frequency_error_hz: Optional[float] = None
    frequency_error_percent: Optional[float] = None
    similarity_score: Optional[float] = None
    message: str = ""
    figure_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class BatchConfig:
    """Zusätzliche Konfiguration des Achsenvergleichs."""

    video_path: Path
    imu_path: Path
    output_dir: Path
    video_time_column: str = "time_seconds"
    video_value_column: str = "inner_pipe_track_center_x"
    imu_time_column: str = "time_seconds"
    axes: tuple[str, ...] = DEFAULT_AXES
    video_name: str = "video"
    imu_name_prefix: str = "imu"
    video_unit: Optional[str] = None
    imu_unit: Optional[str] = None
    normalize: str = "zscore"
    detrend: bool = True
    center: bool = True
    smoothing_window: int = 1
    lowpass_hz: Optional[float] = None
    highpass_hz: Optional[float] = None
    synchronization_method: str = "cross_correlation"
    max_lag_s: Optional[float] = None
    manual_lag_s: Optional[float] = None
    target_sample_rate_hz: Optional[float] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    save_intermediate: bool = True
    create_plots: bool = True
    create_report: bool = True
    overwrite: bool = True
    keep_prepared_imu: bool = True

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path).expanduser()
        self.imu_path = Path(self.imu_path).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.axes = tuple(dict.fromkeys(self.axes))
        if not self.axes:
            raise ValueError("Mindestens eine IMU-Achse muss angegeben werden.")


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_table(path: Path) -> pd.DataFrame:
    """Liest die im Repository verwendeten CSV/TSV-Dateien robust."""

    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    )
    last_error: Optional[Exception] = None
    for options in attempts:
        try:
            frame = pd.read_csv(path, **options)
            if len(frame.columns) >= 1:
                return frame
        except Exception as exc:  # pragma: no cover - abhängig von Eingabedatei
            last_error = exc
    raise RuntimeError(f"Datei konnte nicht gelesen werden: {path}") from last_error


def _parse_imu_time(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Erzeugt eine streng numerische relative Zeitachse in Sekunden.

    Priorität:
    1. vorhandenes ``time_seconds``;
    2. Kombination aus ``day`` und ``time``;
    3. numerische Einzelzeitspalten;
    4. parsebare Zeitstempelspalten.
    """

    if "time_seconds" in frame.columns:
        numeric = pd.to_numeric(frame["time_seconds"], errors="coerce")
        if numeric.notna().sum() >= 2:
            values = numeric.to_numpy(float)
            return values - values[np.flatnonzero(np.isfinite(values))[0]], "time_seconds"

    if {"day", "time"}.issubset(frame.columns):
        combined = frame["day"].astype(str).str.strip() + " " + frame["time"].astype(str).str.strip()
        # Das Archiv enthält üblicherweise Mikrosekunden. Fallback ohne festes
        # Format deckt Dateien ohne Nachkommastellen ab.
        timestamps = pd.to_datetime(
            combined,
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
        if timestamps.notna().sum() < 2:
            timestamps = pd.to_datetime(combined, errors="coerce")
        if timestamps.notna().sum() >= 2:
            first = timestamps.dropna().iloc[0]
            return (timestamps - first).dt.total_seconds().to_numpy(float), "day+time"

    for column in ("time", "timestamp", "writetime", "t"):
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() >= 2:
            values = numeric.to_numpy(float)
            return values - values[np.flatnonzero(np.isfinite(values))[0]], column
        timestamps = pd.to_datetime(frame[column], errors="coerce")
        if timestamps.notna().sum() >= 2:
            first = timestamps.dropna().iloc[0]
            return (timestamps - first).dt.total_seconds().to_numpy(float), column

    raise ValueError(
        "Keine interpretierbare IMU-Zeit gefunden. Erwartet werden "
        "time_seconds, day+time, time, timestamp, writetime oder t."
    )


def prepare_imu_csv(source: Path, destination: Path) -> tuple[Path, dict[str, Any]]:
    """Schreibt eine Validierungs-CSV mit sauberer ``time_seconds``-Spalte."""

    frame = _read_table(source)
    time_seconds, source_column = _parse_imu_time(frame)
    prepared = frame.copy()
    if "time_seconds" in prepared.columns:
        prepared["time_seconds"] = time_seconds
    else:
        insertion = min(2, len(prepared.columns))
        prepared.insert(insertion, "time_seconds", time_seconds)

    finite = np.isfinite(prepared["time_seconds"].to_numpy(float))
    removed = int((~finite).sum())
    prepared = prepared.loc[finite].copy()
    prepared = prepared.sort_values("time_seconds", kind="stable")
    prepared = prepared.drop_duplicates(subset=["time_seconds"], keep="first")

    if len(prepared) < 2:
        raise ValueError("Nach Zeitaufbereitung bleiben weniger als zwei IMU-Zeilen.")
    if np.any(np.diff(prepared["time_seconds"].to_numpy(float)) <= 0):
        raise ValueError("Die aufbereitete IMU-Zeit ist nicht streng monoton.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(destination, index=False)
    dt = np.diff(prepared["time_seconds"].to_numpy(float))
    metadata = {
        "source": str(source),
        "prepared_csv": str(destination),
        "time_source": source_column,
        "input_rows": int(len(frame)),
        "output_rows": int(len(prepared)),
        "removed_invalid_time_rows": removed,
        "duration_s": float(prepared["time_seconds"].iloc[-1] - prepared["time_seconds"].iloc[0]),
        "sample_rate_hz": float(1.0 / np.median(dt)),
    }
    return destination, metadata


def _dominant_frequency(time_s: np.ndarray, values: np.ndarray) -> Optional[float]:
    """Bestimmt die stärkste positive FFT-Frequenz ohne DC-Anteil."""

    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(time_s) & np.isfinite(values)
    time_s, values = time_s[mask], values[mask]
    if len(values) < 8:
        return None
    dt = float(np.median(np.diff(time_s)))
    if not math.isfinite(dt) or dt <= 0:
        return None
    values = values - np.mean(values)
    window = np.hanning(len(values))
    spectrum = np.abs(np.fft.rfft(values * window))
    frequencies = np.fft.rfftfreq(len(values), d=dt)
    if len(spectrum) <= 1:
        return None
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    return _finite_float(frequencies[index])


def _spectral_metrics(axis_dir: Path, video_name: str, imu_name: str) -> dict[str, Optional[float]]:
    aligned_path = axis_dir / "data" / "aligned_signals.csv"
    if not aligned_path.is_file():
        return {}
    frame = pd.read_csv(aligned_path)
    if "time_s" not in frame.columns:
        return {}
    signal_columns = [column for column in frame.columns if column not in {"time_s", "residual"}]
    if len(signal_columns) < 2:
        return {}
    video_column = video_name if video_name in frame.columns else signal_columns[0]
    imu_column = imu_name if imu_name in frame.columns else signal_columns[1]
    video_freq = _dominant_frequency(frame["time_s"].to_numpy(float), frame[video_column].to_numpy(float))
    imu_freq = _dominant_frequency(frame["time_s"].to_numpy(float), frame[imu_column].to_numpy(float))
    error = abs(video_freq - imu_freq) if video_freq is not None and imu_freq is not None else None
    percent = (
        100.0 * error / abs(imu_freq)
        if error is not None and imu_freq not in (None, 0.0)
        else None
    )
    return {
        "video_dominant_frequency_hz": video_freq,
        "imu_dominant_frequency_hz": imu_freq,
        "frequency_error_hz": error,
        "frequency_error_percent": percent,
    }


def _similarity_score(correlation: Optional[float], frequency_error_percent: Optional[float]) -> Optional[float]:
    """Heuristischer Ranking-Score; kein zusätzlicher Validierungsnachweis."""

    if correlation is None:
        return None
    correlation_component = min(abs(correlation), 1.0)
    if frequency_error_percent is None:
        frequency_component = 0.0
    else:
        frequency_component = math.exp(-max(frequency_error_percent, 0.0) / 25.0)
    return float(0.7 * correlation_component + 0.3 * frequency_component)


def _axis_config(batch: BatchConfig, prepared_imu: Path, axis: str) -> ValidationConfig:
    return ValidationConfig(
        video_path=batch.video_path,
        imu_path=prepared_imu,
        output_dir=batch.output_dir / axis,
        video_time_column=batch.video_time_column,
        video_value_column=batch.video_value_column,
        imu_time_column=batch.imu_time_column,
        imu_value_column=axis,
        video_name=batch.video_name,
        imu_name=f"{batch.imu_name_prefix} {axis}",
        video_unit=batch.video_unit,
        imu_unit=batch.imu_unit,
        normalize=batch.normalize,
        detrend=batch.detrend,
        center=batch.center,
        smoothing_window=batch.smoothing_window,
        lowpass_hz=batch.lowpass_hz,
        highpass_hz=batch.highpass_hz,
        synchronization_method=batch.synchronization_method,
        max_lag_s=batch.max_lag_s,
        manual_lag_s=batch.manual_lag_s,
        target_sample_rate_hz=batch.target_sample_rate_hz,
        start_time_s=batch.start_time_s,
        end_time_s=batch.end_time_s,
        save_intermediate=batch.save_intermediate,
        create_plots=batch.create_plots,
        create_report=batch.create_report,
        overwrite=batch.overwrite,
        run_name=f"{batch.video_path.stem}_{axis}",
        extra={"batch_axis": axis, "orchestrator": "validate_video_vs_imu_v3"},
    )


def _result_row(axis: str, result: ValidationResult, video_name: str, imu_name: str) -> AxisResult:
    metrics = result.metrics
    spectral = _spectral_metrics(result.output_dir, video_name, imu_name)
    pearson = _finite_float(metrics.get("pearson_correlation"))
    frequency_error_percent = _finite_float(spectral.get("frequency_error_percent"))
    return AxisResult(
        axis=axis,
        status="success",
        output_dir=str(result.output_dir),
        pearson_correlation=pearson,
        absolute_correlation=abs(pearson) if pearson is not None else None,
        correlation_at_lag=_finite_float(metrics.get("correlation_at_lag")),
        lag_s=_finite_float(metrics.get("estimated_lag_s")),
        rmse=_finite_float(metrics.get("rmse")),
        mae=_finite_float(metrics.get("mae")),
        r_squared=_finite_float(metrics.get("r_squared")),
        samples=int(metrics["samples"]) if metrics.get("samples") is not None else None,
        duration_s=_finite_float(metrics.get("duration_s")),
        video_dominant_frequency_hz=_finite_float(spectral.get("video_dominant_frequency_hz")),
        imu_dominant_frequency_hz=_finite_float(spectral.get("imu_dominant_frequency_hz")),
        frequency_error_hz=_finite_float(spectral.get("frequency_error_hz")),
        frequency_error_percent=frequency_error_percent,
        similarity_score=_similarity_score(pearson, frequency_error_percent),
        figure_paths=[str(path) for path in result.figure_paths],
    )


def _comparison_frame(results: Iterable[AxisResult]) -> pd.DataFrame:
    frame = pd.DataFrame([item.as_dict() for item in results])
    if frame.empty:
        return frame
    status_rank = frame["status"].eq("success").astype(int)
    score = pd.to_numeric(frame["similarity_score"], errors="coerce").fillna(-np.inf)
    absolute_corr = pd.to_numeric(frame["absolute_correlation"], errors="coerce").fillna(-np.inf)
    frame = frame.assign(_status=status_rank, _score=score, _corr=absolute_corr)
    frame = frame.sort_values(["_status", "_score", "_corr"], ascending=False, kind="stable")
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame.drop(columns=["_status", "_score", "_corr"])


def _create_summary_plots(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    if frame.empty or not frame["status"].eq("success").any():
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("Matplotlib fehlt; keine Batch-Übersichtsplots.")
        return []

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    successful = frame.loc[frame["status"].eq("success")].copy()
    paths: list[str] = []

    path = plot_dir / "axis_correlation_ranking.png"
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(successful["axis"], successful["absolute_correlation"])
    axis.set_xlabel("IMU-Achse")
    axis.set_ylabel("|Pearson-Korrelation|")
    axis.set_title("Vergleich der IMU-Achsen")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    if successful["frequency_error_percent"].notna().any():
        path = plot_dir / "axis_frequency_error.png"
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(successful["axis"], successful["frequency_error_percent"])
        axis.set_xlabel("IMU-Achse")
        axis.set_ylabel("Frequenzfehler [%]")
        axis.set_title("Dominante Frequenz: Video vs. IMU")
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path))

    return paths


def _scientific_assessment(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame.loc[frame["status"].eq("success")].copy()
    if successful.empty:
        return {
            "classification": "validation_failed",
            "message": "Keine IMU-Achse konnte erfolgreich ausgewertet werden.",
            "recommendations": ["Fehlermeldungen und Eingabespalten prüfen."],
        }

    best = successful.iloc[0]
    best_corr = _finite_float(best.get("absolute_correlation")) or 0.0
    best_freq_error = _finite_float(best.get("frequency_error_percent"))
    if best_corr >= 0.7:
        classification = "strong_time_domain_similarity"
        message = "Mindestens eine Achse zeigt eine starke zeitliche Übereinstimmung."
    elif best_corr >= 0.4:
        classification = "moderate_time_domain_similarity"
        message = "Die beste Achse zeigt eine mittlere zeitliche Übereinstimmung."
    elif best_corr >= 0.2:
        classification = "weak_time_domain_similarity"
        message = "Es ist nur eine schwache zeitliche Übereinstimmung erkennbar."
    else:
        classification = "no_direct_time_domain_similarity"
        message = "Keine Achse zeigt eine belastbare direkte Zeitbereichskorrelation."

    recommendations: list[str] = []
    if best_corr < 0.2:
        recommendations.extend(
            [
                "Die aligned_signals.png der besten Achsen visuell auf dieselbe Schwingungsperiode prüfen.",
                "Position aus dem Video nicht unkommentiert direkt mit linearer Beschleunigung vergleichen.",
                "Video-Geschwindigkeit (erste Ableitung) und Video-Beschleunigung (zweite Ableitung) als physikalisch passendere Größen testen.",
                "IMU-Beschleunigung vor einer Positionsinterpretation filtern, Offset entfernen und nur mit dokumentierten Randbedingungen integrieren.",
                "Sensororientierung und Vorzeichen anhand des Versuchsaufbaus verifizieren.",
            ]
        )
    if best_freq_error is not None and best_freq_error <= 10.0 and best_corr < 0.2:
        recommendations.append(
            "Die dominante Frequenz stimmt trotz niedriger Korrelation ungefähr überein; Phase, Vorzeichen, Drift oder unterschiedliche physikalische Größen untersuchen."
        )
    if best_freq_error is None or best_freq_error > 20.0:
        recommendations.append(
            "Frequenzspektren prüfen; möglicherweise enthalten Video und IMU nicht denselben Messabschnitt oder nicht dieselbe Bewegungskomponente."
        )

    return {
        "classification": classification,
        "message": message,
        "best_axis": str(best["axis"]),
        "best_absolute_correlation": best_corr,
        "best_frequency_error_percent": best_freq_error,
        "recommendations": recommendations,
        "note": "Der similarity_score dient nur dem Ranking und ist kein statistischer Signifikanznachweis.",
    }


def _write_summary_html(path: Path, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    table = frame.drop(columns=["figure_paths"], errors="ignore").to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}")
    assessment = summary["assessment"]
    recommendations = "".join(f"<li>{html.escape(str(item))}</li>" for item in assessment.get("recommendations", []))
    body = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Video–IMU Achsenvergleich</title>
<style>body{{font-family:Arial,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;line-height:1.45}}table{{border-collapse:collapse;width:100%;font-size:.88rem}}th,td{{border:1px solid #ccc;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:center}}th{{background:#eee}}code{{background:#f4f4f4;padding:.1rem .3rem}}.box{{border:1px solid #bbb;padding:1rem;margin:1rem 0}}</style></head>
<body><h1>Video–IMU Achsenvergleich</h1>
<div class="box"><strong>{html.escape(assessment.get('message', ''))}</strong><br>
Beste Achse: <code>{html.escape(str(assessment.get('best_axis', 'n/a')))}</code></div>
<h2>Ranking</h2>{table}
<h2>Empfohlene nächste Prüfungen</h2><ul>{recommendations}</ul>
<p><small>Automatisch erzeugt mit validate_video_vs_imu_v3.py.</small></p></body></html>"""
    path.write_text(body, encoding="utf-8")


def run_axis_comparison(batch: BatchConfig) -> dict[str, Any]:
    """Bereitet die IMU auf und führt die v2-Pipeline für alle Achsen aus."""

    started = perf_counter()
    if not batch.video_path.is_file():
        raise FileNotFoundError(batch.video_path)
    if not batch.imu_path.is_file():
        raise FileNotFoundError(batch.imu_path)
    batch.output_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = batch.output_dir / "prepared" / "imu_validation_ready.csv"
    prepared_path, preparation = prepare_imu_csv(batch.imu_path, prepared_path)
    prepared_frame = pd.read_csv(prepared_path, nrows=3)
    available_axes = [axis for axis in batch.axes if axis in prepared_frame.columns]
    missing_axes = [axis for axis in batch.axes if axis not in prepared_frame.columns]
    if missing_axes:
        LOGGER.warning("IMU-Achsen fehlen und werden übersprungen: %s", ", ".join(missing_axes))
    if not available_axes:
        raise ValueError(f"Keine gewünschte IMU-Achse vorhanden. Spalten: {list(prepared_frame.columns)}")

    results: list[AxisResult] = []
    for number, axis in enumerate(available_axes, start=1):
        LOGGER.info("Achse %d/%d: %s", number, len(available_axes), axis)
        imu_name = f"{batch.imu_name_prefix} {axis}"
        try:
            validation_result = validate_video_vs_imu(_axis_config(batch, prepared_path, axis))
            results.append(_result_row(axis, validation_result, batch.video_name, imu_name))
        except Exception as exc:
            LOGGER.error("Achse %s fehlgeschlagen: %s", axis, exc)
            results.append(
                AxisResult(axis=axis, status="failed", output_dir=str(batch.output_dir / axis), message=str(exc))
            )

    for axis in missing_axes:
        results.append(
            AxisResult(axis=axis, status="missing", output_dir=str(batch.output_dir / axis), message="Spalte fehlt in der IMU-Datei.")
        )

    comparison = _comparison_frame(results)
    comparison_path = batch.output_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    plots = _create_summary_plots(comparison, batch.output_dir) if batch.create_plots else []
    assessment = _scientific_assessment(comparison)
    summary = {
        "video_path": str(batch.video_path),
        "imu_path": str(batch.imu_path),
        "output_dir": str(batch.output_dir.resolve()),
        "axes_requested": list(batch.axes),
        "axes_processed": available_axes,
        "imu_preparation": preparation,
        "assessment": assessment,
        "comparison_csv": str(comparison_path.resolve()),
        "summary_plots": plots,
        "runtime_s": perf_counter() - started,
        "configuration": dataclasses.asdict(batch),
    }
    summary_path = batch.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    _write_summary_html(batch.output_dir / "summary.html", comparison, summary)

    if not batch.keep_prepared_imu:
        shutil.rmtree(prepared_path.parent, ignore_errors=True)

    LOGGER.info("Achsenvergleich abgeschlossen: %s", comparison_path.resolve())
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Robuster Video–IMU-Achsenvergleich auf Basis von validate_video_vs_imu_v2.py.",
    )
    parser.add_argument("--video-csv", "--video-path", dest="video_path", type=Path, required=True)
    parser.add_argument("--imu-csv", "--imu-path", dest="imu_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-time-column", default="time_seconds")
    parser.add_argument("--video-value-column", default="inner_pipe_track_center_x")
    parser.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES))
    parser.add_argument("--video-name", default="video")
    parser.add_argument("--imu-name-prefix", default="imu")
    parser.add_argument("--video-unit")
    parser.add_argument("--imu-unit")
    parser.add_argument("--normalize", choices=["none", "zscore", "minmax", "robust"], default="zscore")
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--lowpass-hz", type=float)
    parser.add_argument("--highpass-hz", type=float)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--synchronization-method", choices=["cross_correlation", "timestamp", "none"], default="cross_correlation")
    parser.add_argument("--max-lag-s", type=float)
    parser.add_argument("--manual-lag-s", type=float)
    parser.add_argument("--target-sample-rate-hz", type=float)
    parser.add_argument("--start-time-s", type=float)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument("--no-intermediate", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--discard-prepared-imu", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        config = BatchConfig(
            video_path=args.video_path,
            imu_path=args.imu_path,
            output_dir=args.output_dir,
            video_time_column=args.video_time_column,
            video_value_column=args.video_value_column,
            axes=tuple(args.axes),
            video_name=args.video_name,
            imu_name_prefix=args.imu_name_prefix,
            video_unit=args.video_unit,
            imu_unit=args.imu_unit,
            normalize=args.normalize,
            detrend=not args.no_detrend,
            center=not args.no_center,
            smoothing_window=args.smoothing_window,
            lowpass_hz=args.lowpass_hz,
            highpass_hz=args.highpass_hz,
            synchronization_method=args.synchronization_method,
            max_lag_s=args.max_lag_s,
            manual_lag_s=args.manual_lag_s,
            target_sample_rate_hz=args.target_sample_rate_hz,
            start_time_s=args.start_time_s,
            end_time_s=args.end_time_s,
            save_intermediate=not args.no_intermediate,
            create_plots=not args.no_plots,
            create_report=not args.no_report,
            keep_prepared_imu=not args.discard_prepared_imu,
        )
        summary = run_axis_comparison(config)
    except KeyboardInterrupt:
        LOGGER.error("Abgebrochen.")
        return 130
    except Exception as exc:
        LOGGER.error("Validierung fehlgeschlagen: %s", exc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
