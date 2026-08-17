#!/usr/bin/env python3
"""
validate_video_vs_imu_v2.py
===========================

End-to-End-Orchestrierung zur Validierung eines aus Video-Tracking gewonnenen
Signals gegenüber einer IMU-Referenz.

Vorgesehene Projektmodule
-------------------------
    video_loader.py
    imu_loader.py
    preprocessing.py
    synchronization.py
    metrics.py
    plotting.py
    report.py

Die Datei ist bewusst modular aufgebaut. Sie unterstützt zwei Betriebsarten:

1. Direkte Verwendung der bekannten Projektmodule über deren öffentliche
   Funktionen.
2. Robuste CSV-Fallbacks, falls Loader oder einzelne Hilfsfunktionen andere
   Namen besitzen oder nicht alle optionalen Funktionen bereitstellen.

Die eigentliche Signalverarbeitung bleibt in den Fachmodulen. Diese Datei
übernimmt Konfiguration, Datenfluss, Schnittstellenanpassung, Validierung,
Export, Logging und Fehlerbehandlung.

Beispiel
--------
python validate_video_vs_imu_v2.py \
    --video-csv outputs/GX010044_results_inner_pipe_track_timeseries.csv \
    --imu-csv data/measured_data/2022-01-28/.../data.csv \
    --video-time-column time_s \
    --video-value-column displacement_y_px \
    --imu-time-column time_s \
    --imu-value-column Acceleration_z \
    --output-dir outputs/validation/GX010044

Alternativ kann eine JSON- oder YAML-Konfiguration übergeben werden:

python validate_video_vs_imu_v2.py --config config/validation.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import logging
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence
from validation import (
    load_video_signal,
    load_imu_signal,
    preprocess_signal,
    synchronize_signals,
    compute_validation_metrics,
    create_validation_plots,
    generate_report,
)

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("NumPy wird für die Validierung benötigt.") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas wird für die Validierung benötigt.") from exc

LOGGER = logging.getLogger("validate_video_vs_imu_v2")


# =============================================================================
# Datenklassen
# =============================================================================

@dataclass(slots=True)
class SignalData:
    """Ein eindimensionales Signal mit Zeitachse und Metadaten."""

    time: np.ndarray
    values: np.ndarray
    name: str
    unit: Optional[str] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=float).reshape(-1)

        if self.time.size != self.values.size:
            raise ValueError(
                f"Zeit- und Signalvektor von '{self.name}' sind unterschiedlich "
                f"lang: {self.time.size} != {self.values.size}."
            )
        if self.time.size < 2:
            raise ValueError(f"Signal '{self.name}' enthält weniger als 2 Werte.")

        finite = np.isfinite(self.time) & np.isfinite(self.values)
        self.time = self.time[finite]
        self.values = self.values[finite]

        if self.time.size < 2:
            raise ValueError(
                f"Signal '{self.name}' enthält nach NaN/Inf-Bereinigung "
                "weniger als 2 Werte."
            )

        order = np.argsort(self.time, kind="stable")
        self.time = self.time[order]
        self.values = self.values[order]

        unique_time, unique_indices = np.unique(self.time, return_index=True)
        self.time = unique_time
        self.values = self.values[unique_indices]

        if self.time.size < 2 or np.any(np.diff(self.time) <= 0):
            raise ValueError(
                f"Zeitachse von '{self.name}' muss streng monoton steigen."
            )

    @property
    def duration_s(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def sample_rate_hz(self) -> float:
        differences = np.diff(self.time)
        median_dt = float(np.median(differences))
        return 1.0 / median_dt if median_dt > 0 else float("nan")

    def copy(self, **changes: Any) -> "SignalData":
        data = dataclasses.asdict(self)
        data.update(changes)
        return SignalData(**data)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time_s": self.time, self.name: self.values})

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "source": self.source,
            "samples": int(self.time.size),
            "start_time_s": float(self.time[0]),
            "end_time_s": float(self.time[-1]),
            "duration_s": self.duration_s,
            "sample_rate_hz": self.sample_rate_hz,
            "minimum": float(np.min(self.values)),
            "maximum": float(np.max(self.values)),
            "mean": float(np.mean(self.values)),
            "standard_deviation": float(np.std(self.values)),
            **self.metadata,
        }


@dataclass(slots=True)
class ValidationConfig:
    """Vollständige Konfiguration der Validierung."""

    video_path: Path
    imu_path: Path
    output_dir: Path

    video_time_column: Optional[str] = None
    video_value_column: Optional[str] = None
    imu_time_column: Optional[str] = None
    imu_value_column: Optional[str] = None

    video_fps: Optional[float] = None
    imu_sample_rate_hz: Optional[float] = None

    video_scale: float = 1.0
    video_offset: float = 0.0
    imu_scale: float = 1.0
    imu_offset: float = 0.0

    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None

    detrend: bool = True
    center: bool = True
    normalize: str = "zscore"
    smoothing_window: int = 1
    lowpass_hz: Optional[float] = None
    highpass_hz: Optional[float] = None

    synchronization_method: str = "cross_correlation"
    max_lag_s: Optional[float] = None
    manual_lag_s: Optional[float] = None
    target_sample_rate_hz: Optional[float] = None

    video_name: str = "video"
    imu_name: str = "imu"
    video_unit: Optional[str] = None
    imu_unit: Optional[str] = None

    report_title: str = "Validierung Video-Tracking vs. IMU"
    run_name: Optional[str] = None
    author: Optional[str] = None

    save_intermediate: bool = True
    create_plots: bool = True
    create_report: bool = True
    overwrite: bool = True
    strict_modules: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path).expanduser()
        self.imu_path = Path(self.imu_path).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()

        if self.video_fps is not None and self.video_fps <= 0:
            raise ValueError("video_fps muss > 0 sein.")
        if self.imu_sample_rate_hz is not None and self.imu_sample_rate_hz <= 0:
            raise ValueError("imu_sample_rate_hz muss > 0 sein.")
        if self.target_sample_rate_hz is not None and self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz muss > 0 sein.")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window muss >= 1 sein.")
        if self.end_time_s is not None and self.start_time_s is not None:
            if self.end_time_s <= self.start_time_s:
                raise ValueError("end_time_s muss größer als start_time_s sein.")
        if self.normalize not in {"none", "zscore", "minmax", "robust"}:
            raise ValueError(
                "normalize muss none, zscore, minmax oder robust sein."
            )


@dataclass(slots=True)
class ValidationResult:
    """Ergebnisobjekt eines vollständigen Validierungslaufs."""

    output_dir: Path
    metrics: dict[str, Any]
    synchronization: dict[str, Any]
    video_summary: dict[str, Any]
    imu_summary: dict[str, Any]
    figure_paths: list[Path] = field(default_factory=list)
    report_artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    runtime_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "metrics": _jsonable(self.metrics),
            "synchronization": _jsonable(self.synchronization),
            "video_summary": _jsonable(self.video_summary),
            "imu_summary": _jsonable(self.imu_summary),
            "figure_paths": [str(path) for path in self.figure_paths],
            "report_artifacts": _jsonable(self.report_artifacts),
            "warnings": list(self.warnings),
            "runtime_s": float(self.runtime_s),
        }


# =============================================================================
# Allgemeine Hilfsfunktionen
# =============================================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "as_dict"):
        try:
            return _jsonable(value.as_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(data), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _import_optional_module(name: str, strict: bool = False) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Modul '{name}' konnte nicht importiert werden.") from exc
        LOGGER.warning("Modul '%s' nicht verfügbar: %s", name, exc)
        return None


def _find_callable(module: Any, names: Sequence[str]) -> Optional[Callable[..., Any]]:
    if module is None:
        return None
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def _call_supported(function: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Ruft eine Funktion nur mit den von ihrer Signatur akzeptierten Parametern auf.
    Dadurch bleiben kleine Schnittstellenunterschiede zwischen Modulversionen
    beherrschbar.
    """

    signature = inspect.signature(function)
    parameters = signature.parameters

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return function(**kwargs)

    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }

    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in accepted
    ]

    if missing:
        raise TypeError(
            f"{function.__module__}.{function.__name__} benötigt nicht "
            f"bereitgestellte Parameter: {missing}"
        )

    return function(**accepted)


def _read_delimited_file(path: Path) -> pd.DataFrame:
    """Liest CSV/TSV/semicolon-Dateien mit robuster Trennzeichenerkennung."""

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json"}:
        data = pd.read_json(path)
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if suffix in {".xlsx", ".xls", ".ods"}:
        return pd.read_excel(path)

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]
    last_error: Optional[Exception] = None
    for options in attempts:
        try:
            frame = pd.read_csv(path, **options)
            if frame.shape[1] >= 1:
                return frame
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Datei konnte nicht gelesen werden: {path}") from last_error


def _select_numeric_column(
    frame: pd.DataFrame,
    requested: Optional[str],
    excluded: Iterable[str] = (),
) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(
                f"Spalte '{requested}' fehlt. Verfügbar: {list(frame.columns)}"
            )
        return requested

    excluded_set = set(excluded)
    candidates = [
        column
        for column in frame.columns
        if column not in excluded_set
        and pd.api.types.is_numeric_dtype(
            pd.to_numeric(frame[column], errors="coerce")
        )
        and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
    ]
    if not candidates:
        raise ValueError("Keine geeignete numerische Signalspalte gefunden.")
    return str(candidates[0])


def _infer_time(
    frame: pd.DataFrame,
    requested_column: Optional[str],
    sample_rate_hz: Optional[float],
) -> tuple[np.ndarray, Optional[str]]:
    if requested_column:
        if requested_column not in frame.columns:
            raise KeyError(
                f"Zeitspalte '{requested_column}' fehlt. "
                f"Verfügbar: {list(frame.columns)}"
            )
        series = frame[requested_column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= 2:
            return numeric.to_numpy(dtype=float), requested_column

        datetimes = pd.to_datetime(series, errors="coerce", utc=True)
        if datetimes.notna().sum() >= 2:
            seconds = (datetimes - datetimes.iloc[0]).dt.total_seconds()
            return seconds.to_numpy(dtype=float), requested_column
        raise ValueError(f"Zeitspalte '{requested_column}' ist nicht interpretierbar.")

    preferred = (
        "time_s", "time", "timestamp_s", "timestamp", "t",
        "seconds", "elapsed_time_s", "relative_time_s",
    )
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for name in preferred:
        if name in lowered:
            column = lowered[name]
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().sum() >= 2:
                return numeric.to_numpy(dtype=float), column
            datetimes = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if datetimes.notna().sum() >= 2:
                seconds = (datetimes - datetimes.iloc[0]).dt.total_seconds()
                return seconds.to_numpy(dtype=float), column

    frame_columns = {str(column).lower(): str(column) for column in frame.columns}
    for index_name in ("frame", "frame_idx", "frame_index", "frame_id"):
        if index_name in frame_columns and sample_rate_hz:
            column = frame_columns[index_name]
            indices = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            return indices / float(sample_rate_hz), column

    if sample_rate_hz:
        return np.arange(len(frame), dtype=float) / float(sample_rate_hz), None

    raise ValueError(
        "Keine Zeitachse erkannt. Bitte Zeitspalte oder Abtastrate angeben."
    )


def _coerce_signal(
    result: Any,
    *,
    name: str,
    unit: Optional[str],
    source: Path,
    time_column: Optional[str],
    value_column: Optional[str],
    sample_rate_hz: Optional[float],
) -> SignalData:
    """Konvertiert verbreitete Loader-Ausgaben in SignalData."""

    if isinstance(result, SignalData):
        return result

    if isinstance(result, pd.DataFrame):
        frame = result
    elif isinstance(result, Mapping):
        time_keys = ("time", "time_s", "timestamps", "timestamp", "t")
        value_keys = (
            value_column, "values", "signal", "data", "displacement",
            "acceleration", name,
        )
        time_value = next(
            (result[key] for key in time_keys if key and key in result), None
        )
        signal_value = next(
            (result[key] for key in value_keys if key and key in result), None
        )
        if time_value is not None and signal_value is not None:
            return SignalData(
                time=np.asarray(time_value),
                values=np.asarray(signal_value),
                name=name,
                unit=unit,
                source=str(source),
                metadata={
                    key: _jsonable(value)
                    for key, value in result.items()
                    if key not in time_keys and key not in value_keys
                },
            )
        frame = pd.DataFrame(result)
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        return SignalData(
            time=np.asarray(result[0]),
            values=np.asarray(result[1]),
            name=name,
            unit=unit,
            source=str(source),
        )
    elif hasattr(result, "time") and hasattr(result, "values"):
        return SignalData(
            time=np.asarray(result.time),
            values=np.asarray(result.values),
            name=name,
            unit=unit,
            source=str(source),
            metadata=_jsonable(getattr(result, "metadata", {})),
        )
    else:
        raise TypeError(
            f"Loader-Ausgabe für '{name}' kann nicht interpretiert werden: "
            f"{type(result).__name__}"
        )

    time, inferred_time_column = _infer_time(
        frame, time_column, sample_rate_hz
    )
    signal_column = _select_numeric_column(
        frame,
        value_column,
        excluded=([inferred_time_column] if inferred_time_column else []),
    )
    values = pd.to_numeric(frame[signal_column], errors="coerce").to_numpy(float)

    return SignalData(
        time=time,
        values=values,
        name=name,
        unit=unit,
        source=str(source),
        metadata={
            "time_column": inferred_time_column,
            "value_column": signal_column,
            "input_rows": int(len(frame)),
        },
    )


# =============================================================================
# Laden
# =============================================================================

def load_video_signal(
    config: ValidationConfig,
    module: Any,
) -> SignalData:
    function = _find_callable(
        module,
        (
            "load_video_signal",
            "load_tracking_signal",
            "load_video_data",
            "load_tracking_data",
            "load_video_csv",
            "load",
        ),
    )

    if function is not None:
        try:
            result = _call_supported(
                function,

                # Actual API of validation.video_loader
                csv_file=config.video_path,
                signal_column=config.video_value_column,
                valid_only=True,

                # Compatibility aliases for other loader versions
                path=config.video_path,
                video_path=config.video_path,
                csv_path=config.video_path,
                file_path=config.video_path,
                time_column=config.video_time_column,
                value_column=config.video_value_column,
                fps=config.video_fps,
                sample_rate_hz=config.video_fps,
            )

            signal = _coerce_signal(
                result,
                name=config.video_name,
                unit=config.video_unit,
                source=config.video_path,
                time_column=config.video_time_column,
                value_column=config.video_value_column,
                sample_rate_hz=config.video_fps,
            )

            signal.metadata["loader"] = (
                f"{function.__module__}.{function.__name__}"
            )

            return signal

        except Exception as exc:
            if config.strict_modules:
                raise

            LOGGER.warning(
                "video_loader-Funktion fehlgeschlagen; "
                "CSV-Fallback wird genutzt: %s",
                exc,
            )

    frame = _read_delimited_file(
        config.video_path
    )

    signal = _coerce_signal(
        frame,
        name=config.video_name,
        unit=config.video_unit,
        source=config.video_path,
        time_column=config.video_time_column,
        value_column=config.video_value_column,
        sample_rate_hz=config.video_fps,
    )

    signal.metadata["loader"] = (
        "internal_csv_fallback"
    )

    return signal

def load_imu_signal(
    config: ValidationConfig,
    module: Any,
) -> SignalData:
    function = _find_callable(
        module,
        (
            "load_imu_signal",
            "load_imu_data",
            "load_measurement_data",
            "load_sensor_data",
            "load_imu_csv",
            "load",
        ),
    )

    if function is not None:
        try:
            result = _call_supported(
                function,

                # Actual API of validation.imu_loader
                csv_file=config.imu_path,
                axis=config.imu_value_column,

                # Compatibility aliases
                path=config.imu_path,
                imu_path=config.imu_path,
                csv_path=config.imu_path,
                file_path=config.imu_path,
                time_column=config.imu_time_column,
                value_column=config.imu_value_column,
                signal_column=config.imu_value_column,
                sample_rate_hz=config.imu_sample_rate_hz,
                frequency_hz=config.imu_sample_rate_hz,
            )

            signal = _coerce_signal(
                result,
                name=config.imu_name,
                unit=config.imu_unit,
                source=config.imu_path,
                time_column=config.imu_time_column,
                value_column=config.imu_value_column,
                sample_rate_hz=config.imu_sample_rate_hz,
            )

            signal.metadata["loader"] = (
                f"{function.__module__}.{function.__name__}"
            )

            return signal

        except Exception as exc:
            if config.strict_modules:
                raise

            LOGGER.warning(
                "imu_loader-Funktion fehlgeschlagen; "
                "CSV-Fallback wird genutzt: %s",
                exc,
            )

    frame = _read_delimited_file(
        config.imu_path
    )

    signal = _coerce_signal(
        frame,
        name=config.imu_name,
        unit=config.imu_unit,
        source=config.imu_path,
        time_column=config.imu_time_column,
        value_column=config.imu_value_column,
        sample_rate_hz=config.imu_sample_rate_hz,
    )

    signal.metadata["loader"] = (
        "internal_csv_fallback"
    )

    return signal


# =============================================================================
# Vorverarbeitung
# =============================================================================

def _crop_signal(
    signal: SignalData,
    start_time_s: Optional[float],
    end_time_s: Optional[float],
) -> SignalData:
    mask = np.ones(signal.time.size, dtype=bool)
    if start_time_s is not None:
        mask &= signal.time >= start_time_s
    if end_time_s is not None:
        mask &= signal.time <= end_time_s
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"Zeitfenster entfernt zu viele Werte aus '{signal.name}'."
        )
    return signal.copy(time=signal.time[mask], values=signal.values[mask])


def _internal_preprocess(
    signal: SignalData,
    *,
    scale: float,
    offset: float,
    config: ValidationConfig,
) -> SignalData:
    time = signal.time.copy()
    values = signal.values.astype(float, copy=True) * scale + offset

    if config.detrend and values.size >= 2:
        coefficients = np.polyfit(time - time[0], values, deg=1)
        values = values - np.polyval(coefficients, time - time[0])

    if config.smoothing_window > 1:
        values = (
            pd.Series(values)
            .rolling(
                window=config.smoothing_window,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy(float)
        )

    if config.center:
        values = values - float(np.mean(values))

    if config.normalize == "zscore":
        std = float(np.std(values))
        if std > np.finfo(float).eps:
            values = values / std
    elif config.normalize == "minmax":
        minimum, maximum = float(np.min(values)), float(np.max(values))
        span = maximum - minimum
        if span > np.finfo(float).eps:
            values = (values - minimum) / span
    elif config.normalize == "robust":
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        iqr = float(q75 - q25)
        values = values - median
        if iqr > np.finfo(float).eps:
            values = values / iqr

    return signal.copy(
        time=time,
        values=values,
        metadata={
            **signal.metadata,
            "preprocessing": {
                "scale": scale,
                "offset": offset,
                "detrend": config.detrend,
                "center": config.center,
                "normalize": config.normalize,
                "smoothing_window": config.smoothing_window,
                "lowpass_hz": config.lowpass_hz,
                "highpass_hz": config.highpass_hz,
                "implementation": "internal_fallback",
            },
        },
    )


def preprocess_signal(
    signal: SignalData,
    *,
    kind: str,
    scale: float,
    offset: float,
    config: ValidationConfig,
    module: Any,
) -> SignalData:
    signal = _crop_signal(
        signal,
        config.start_time_s,
        config.end_time_s,
    )

    function = _find_callable(
        module,
        (
            f"preprocess_{kind}_signal",
            f"preprocess_{kind}",
            "preprocess_signal",
            "prepare_signal",
            "preprocess",
        ),
    )

    if function is not None:
        try:
            result = _call_supported(
                function,

                # Actual API of validation.preprocessing
                time=signal.time,
                signal=signal.values,
                target_fs=config.target_sample_rate_hz,
                detrend_signal=config.detrend,
                normalization=config.normalize,

                # Compatibility aliases
                values=signal.values,
                data=signal.values,
                sample_rate_hz=signal.sample_rate_hz,
                scale=scale,
                offset=offset,
                detrend=config.detrend,
                center=config.center,
                normalize=config.normalize,
                smoothing_window=config.smoothing_window,
                lowpass_hz=config.lowpass_hz,
                highpass_hz=config.highpass_hz,
            )

            processed = _coerce_signal(
                result,
                name=signal.name,
                unit=signal.unit,
                source=Path(signal.source or kind),
                time_column=None,
                value_column=None,
                sample_rate_hz=signal.sample_rate_hz,
            )

            processed.metadata = {
                **signal.metadata,
                **processed.metadata,
                "preprocessing_function": (
                    f"{function.__module__}.{function.__name__}"
                ),
            }

            return processed

        except Exception as exc:
            if config.strict_modules:
                raise

            LOGGER.warning(
                "preprocessing.%s für %s fehlgeschlagen; "
                "Fallback: %s",
                function.__name__,
                kind,
                exc,
            )

    return _internal_preprocess(
        signal,
        scale=scale,
        offset=offset,
        config=config,
    )

# =============================================================================
# Synchronisation
# =============================================================================

def _common_time_grid(
    video: SignalData,
    imu: SignalData,
    target_rate_hz: Optional[float],
) -> np.ndarray:
    start = max(float(video.time[0]), float(imu.time[0]))
    end = min(float(video.time[-1]), float(imu.time[-1]))
    if end <= start:
        raise ValueError("Video- und IMU-Signal besitzen keine zeitliche Überlappung.")

    rate = target_rate_hz
    if rate is None:
        rates = [
            rate
            for rate in (video.sample_rate_hz, imu.sample_rate_hz)
            if math.isfinite(rate) and rate > 0
        ]
        rate = min(rates) if rates else 100.0

    number = int(math.floor((end - start) * rate)) + 1
    if number < 3:
        raise ValueError("Gemeinsamer Zeitbereich enthält weniger als 3 Abtastwerte.")
    return start + np.arange(number, dtype=float) / rate


def _normalized_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
    max_lag_samples: Optional[int],
) -> tuple[int, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first - np.mean(first)
    second = second - np.mean(second)

    first_std = np.std(first)
    second_std = np.std(second)
    if first_std <= np.finfo(float).eps or second_std <= np.finfo(float).eps:
        raise ValueError("Kreuzkorrelation benötigt Signale mit Varianz.")

    first /= first_std
    second /= second_std

    correlation = np.correlate(first, second, mode="full")
    lags = np.arange(-second.size + 1, first.size)

    if max_lag_samples is not None:
        mask = np.abs(lags) <= max_lag_samples
        correlation = correlation[mask]
        lags = lags[mask]

    normalization = np.minimum(
        first.size,
        second.size,
    )
    correlation = correlation / max(normalization, 1)
    index = int(np.argmax(np.abs(correlation)))
    return int(lags[index]), float(correlation[index])


def _internal_synchronize(
    video: SignalData,
    imu: SignalData,
    config: ValidationConfig,
) -> tuple[SignalData, SignalData, dict[str, Any]]:
    base_grid = _common_time_grid(video, imu, config.target_sample_rate_hz)
    video_base = np.interp(base_grid, video.time, video.values)
    imu_base = np.interp(base_grid, imu.time, imu.values)
    rate = 1.0 / float(np.median(np.diff(base_grid)))

    if config.manual_lag_s is not None:
        lag_s = float(config.manual_lag_s)
        correlation_at_lag = float("nan")
        lag_samples = int(round(lag_s * rate))
    elif config.synchronization_method in {"none", "timestamp", "timestamps"}:
        lag_s = 0.0
        lag_samples = 0
        correlation_at_lag = float(np.corrcoef(video_base, imu_base)[0, 1])
    else:
        max_lag_samples = (
            int(round(config.max_lag_s * rate))
            if config.max_lag_s is not None
            else None
        )
        lag_samples, correlation_at_lag = _normalized_cross_correlation(
            video_base, imu_base, max_lag_samples
        )
        # np.correlate(video, imu): positiver Lag bedeutet, dass das Video
        # gegenüber der IMU nach rechts verschoben ist. Zur Ausrichtung wird
        # die Video-Zeitachse daher um -lag korrigiert.
        lag_s = float(lag_samples / rate)

    shifted_video_time = video.time - lag_s
    aligned_start = max(float(shifted_video_time[0]), float(imu.time[0]))
    aligned_end = min(float(shifted_video_time[-1]), float(imu.time[-1]))
    if aligned_end <= aligned_start:
        raise ValueError("Geschätzter Lag entfernt die gesamte Überlappung.")

    number = int(math.floor((aligned_end - aligned_start) * rate)) + 1
    aligned_time = aligned_start + np.arange(number, dtype=float) / rate

    video_values = np.interp(aligned_time, shifted_video_time, video.values)
    imu_values = np.interp(aligned_time, imu.time, imu.values)
    relative_time = aligned_time - aligned_time[0]

    aligned_video = video.copy(time=relative_time, values=video_values)
    aligned_imu = imu.copy(time=relative_time, values=imu_values)

    synchronization = {
        "method": config.synchronization_method,
        "implementation": "internal_fallback",
        "lag_samples": int(lag_samples),
        "lag_s": float(lag_s),
        "correlation_at_lag": correlation_at_lag,
        "target_sample_rate_hz": float(rate),
        "overlap_start_s": float(aligned_start),
        "overlap_end_s": float(aligned_end),
        "overlap_duration_s": float(aligned_end - aligned_start),
        "aligned_samples": int(relative_time.size),
    }
    return aligned_video, aligned_imu, synchronization


def synchronize_signals(
    video: SignalData,
    imu: SignalData,
    config: ValidationConfig,
    module: Any,
) -> tuple[SignalData, SignalData, dict[str, Any]]:
    function = _find_callable(
        module,
        (
            "synchronize_signals",
            "synchronise_signals",
            "align_signals",
            "synchronize_video_imu",
            "estimate_and_apply_lag",
            "synchronize",
        ),
    )

    if function is not None:
        try:
            sampling_rate = (
                float(config.target_sample_rate_hz)
                if config.target_sample_rate_hz is not None
                else float(
                    min(
                        video.sample_rate_hz,
                        imu.sample_rate_hz,
                    )
                )
            )

            result = _call_supported(
                function,

                # Actual API of validation.synchronization
                video_time=video.time,
                video_signal=video.values,
                imu_time=imu.time,
                imu_signal=imu.values,
                sampling_rate=sampling_rate,

                # Compatibility aliases
                video=video,
                imu=imu,
                video_values=video.values,
                imu_values=imu.values,
                method=config.synchronization_method,
                max_lag_s=config.max_lag_s,
                manual_lag_s=config.manual_lag_s,
                target_sample_rate_hz=(
                    config.target_sample_rate_hz
                ),
            )

            if isinstance(result, Mapping):
                raw_video = (
                    result.get("video")
                    or result.get("video_signal")
                    or result.get("aligned_video")
                )

                raw_imu = (
                    result.get("imu")
                    or result.get("imu_signal")
                    or result.get("aligned_imu")
                )

                metadata = dict(
                    result.get("metadata", result)
                )

            elif (
                isinstance(result, (tuple, list))
                and len(result) >= 2
            ):
                raw_video = result[0]
                raw_imu = result[1]

                metadata = (
                    dict(result[2])
                    if (
                        len(result) >= 3
                        and isinstance(result[2], Mapping)
                    )
                    else {}
                )

            else:
                raise TypeError(
                    "Unbekanntes Synchronisationsergebnis."
                )

            aligned_video = _coerce_signal(
                raw_video,
                name=video.name,
                unit=video.unit,
                source=Path(
                    video.source or "video"
                ),
                time_column=None,
                value_column=None,
                sample_rate_hz=(
                    config.target_sample_rate_hz
                ),
            )

            aligned_imu = _coerce_signal(
                raw_imu,
                name=imu.name,
                unit=imu.unit,
                source=Path(
                    imu.source or "imu"
                ),
                time_column=None,
                value_column=None,
                sample_rate_hz=(
                    config.target_sample_rate_hz
                ),
            )

            metadata["implementation"] = (
                f"{function.__module__}.{function.__name__}"
            )

            return (
                aligned_video,
                aligned_imu,
                _jsonable(metadata),
            )

        except Exception as exc:
            if config.strict_modules:
                raise

            LOGGER.warning(
                "Synchronisationsmodul fehlgeschlagen; "
                "Fallback wird genutzt: %s",
                exc,
            )

    return _internal_synchronize(
        video,
        imu,
        config,
    )


# =============================================================================
# Kennzahlen
# =============================================================================

def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= np.finfo(float).eps or np.std(b) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _internal_metrics(
    video: SignalData,
    imu: SignalData,
) -> dict[str, Any]:
    if video.values.size != imu.values.size:
        raise ValueError("Ausgerichtete Signale müssen gleich lang sein.")

    residual = video.values - imu.values
    absolute = np.abs(residual)
    squared = residual ** 2

    correlation = _safe_correlation(video.values, imu.values)
    rmse = float(np.sqrt(np.mean(squared)))
    mae = float(np.mean(absolute))

    denominator = float(np.sum((imu.values - np.mean(imu.values)) ** 2))
    r_squared = (
        1.0 - float(np.sum(squared)) / denominator
        if denominator > np.finfo(float).eps
        else float("nan")
    )

    return {
        "samples": int(video.values.size),
        "duration_s": float(video.duration_s),
        "sample_rate_hz": float(video.sample_rate_hz),
        "pearson_correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "median_absolute_error": float(np.median(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "bias": float(np.mean(residual)),
        "residual_standard_deviation": float(np.std(residual)),
        "video_mean": float(np.mean(video.values)),
        "video_std": float(np.std(video.values)),
        "imu_mean": float(np.mean(imu.values)),
        "imu_std": float(np.std(imu.values)),
        "implementation": "internal_fallback",
    }


def compute_metrics(
    video: SignalData,
    imu: SignalData,
    synchronization: Mapping[str, Any],
    config: ValidationConfig,
    module: Any,
) -> dict[str, Any]:
    function = _find_callable(
        module,
        (
            "compute_all_metrics",
            "calculate_all_metrics",
            "compute_validation_metrics",
            "calculate_metrics",
            "compute_metrics",
            "evaluate_signals",
            "evaluate",
        ),
    )

    if function is not None:
        try:
            result = _call_supported(
                function,

                # Actual API of validation.metrics
                time=video.time,
                video_signal=video.values,
                imu_signal=imu.values,
                sampling_rate=video.sample_rate_hz,

                # Compatibility aliases
                reference=imu.values,
                prediction=video.values,
                y_true=imu.values,
                y_pred=video.values,
                video_values=video.values,
                imu_values=imu.values,
                sample_rate_hz=video.sample_rate_hz,
                synchronization=synchronization,
            )

            if isinstance(result, Mapping):
                metrics = dict(
                    _jsonable(result)
                )

            elif isinstance(result, pd.Series):
                metrics = dict(
                    _jsonable(
                        result.to_dict()
                    )
                )

            elif isinstance(result, pd.DataFrame):
                if {
                    "metric",
                    "value",
                }.issubset(result.columns):
                    metrics = dict(
                        zip(
                            result["metric"].astype(str),
                            result["value"],
                            strict=False,
                        )
                    )
                else:
                    metrics = {
                        "table": _jsonable(result)
                    }

            elif dataclasses.is_dataclass(result):
                metrics = dict(
                    _jsonable(
                        dataclasses.asdict(result)
                    )
                )

            else:
                metrics = dict(
                    _jsonable(
                        vars(result)
                    )
                )

            metrics.setdefault(
                "implementation",
                f"{function.__module__}.{function.__name__}",
            )

            return metrics

        except Exception as exc:
            if config.strict_modules:
                raise

            LOGGER.warning(
                "metrics-Modul fehlgeschlagen; "
                "Fallback wird genutzt: %s",
                exc,
            )

    return _internal_metrics(
        video,
        imu,
    )


# =============================================================================
# Plotten
# =============================================================================

def _internal_plots(
    video: SignalData,
    imu: SignalData,
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("Matplotlib fehlt; es werden keine Abbildungen erzeugt.")
        return []

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    comparison = plot_dir / "aligned_signals.png"
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(video.time, video.values, label=video.name, linewidth=1.0)
    axis.plot(imu.time, imu.values, label=imu.name, linewidth=1.0, alpha=0.8)
    axis.set_xlabel("Zeit [s]")
    axis.set_ylabel("Normierte Amplitude")
    axis.set_title("Synchronisierte Video- und IMU-Signale")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(comparison, dpi=160)
    plt.close(figure)
    paths.append(comparison)

    scatter = plot_dir / "video_vs_imu_scatter.png"
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(imu.values, video.values, s=7, alpha=0.45)
    axis.set_xlabel(imu.name)
    axis.set_ylabel(video.name)
    axis.set_title("Video gegen IMU")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(scatter, dpi=160)
    plt.close(figure)
    paths.append(scatter)

    residual_path = plot_dir / "residuals.png"
    residual = video.values - imu.values
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(video.time, residual, linewidth=0.9)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Zeit [s]")
    axis.set_ylabel("Video - IMU")
    axis.set_title("Residuen")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(residual_path, dpi=160)
    plt.close(figure)
    paths.append(residual_path)

    return paths


def create_plots(
    video: SignalData,
    imu: SignalData,
    metrics: Mapping[str, Any],
    synchronization: Mapping[str, Any],
    config: ValidationConfig,
    module: Any,
) -> list[Path]:
    if not config.create_plots:
        return []

    plot_dir = config.output_dir / "figures"
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    function = _find_callable(
        module,
        (
            "create_validation_plots",
            "plot_validation_results",
            "plot_video_vs_imu",
            "generate_plots",
            "create_plots",
            "plot_all",
        ),
    )

    if function is not None:
        try:
            result = _call_supported(
                function,

                # Actual API of validation.plotting
                output_dir=plot_dir,
                time=video.time,
                video_signal=video.values,
                imu_signal=imu.values,
                sampling_rate=video.sample_rate_hz,

                # Labels
                video_label=video.name,
                imu_label=imu.name,

                # Compatibility aliases
                video=video,
                imu=imu,
                video_values=video.values,
                imu_values=imu.values,
                metrics=metrics,
                synchronization=synchronization,
                save_dir=plot_dir,
                sample_rate_hz=video.sample_rate_hz,
                show=False,
            )

            if result is None:
                paths = sorted(
                    path
                    for path in plot_dir.iterdir()
                    if path.suffix.lower()
                    in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".svg",
                        ".webp",
                    }
                )

            elif isinstance(result, Mapping):
                paths = [
                    Path(path)
                    for path in result.values()
                ]

            elif isinstance(result, (str, Path)):
                paths = [Path(result)]

            else:
                paths = [
                    Path(path)
                    for path in result
                ]

            paths = [
                path
                for path in paths
                if path.exists()
            ]

            if paths:
                return paths

        except Exception as exc:
            if config.strict_modules:
                raise

            LOGGER.warning(
                "plotting-Modul fehlgeschlagen; "
                "Fallback wird genutzt: %s",
                exc,
            )

    return _internal_plots(
        video,
        imu,
        config.output_dir,
    )


# =============================================================================
# Bericht und Exporte
# =============================================================================

def _save_intermediate(
    output_dir: Path,
    raw_video: SignalData,
    raw_imu: SignalData,
    processed_video: SignalData,
    processed_imu: SignalData,
    aligned_video: SignalData,
    aligned_imu: SignalData,
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_video.as_frame().to_csv(data_dir / "video_raw.csv", index=False)
    raw_imu.as_frame().to_csv(data_dir / "imu_raw.csv", index=False)
    processed_video.as_frame().to_csv(
        data_dir / "video_preprocessed.csv", index=False
    )
    processed_imu.as_frame().to_csv(
        data_dir / "imu_preprocessed.csv", index=False
    )

    aligned = pd.DataFrame(
        {
            "time_s": aligned_video.time,
            aligned_video.name: aligned_video.values,
            aligned_imu.name: aligned_imu.values,
            "residual": aligned_video.values - aligned_imu.values,
        }
    )
    aligned.to_csv(data_dir / "aligned_signals.csv", index=False)


def _generate_report(
    config: ValidationConfig,
    metrics: Mapping[str, Any],
    figures: Sequence[Path],
    video_summary: Mapping[str, Any],
    imu_summary: Mapping[str, Any],
    preprocessing_info: Mapping[str, Any],
    synchronization_info: Mapping[str, Any],
    warnings: Sequence[str],
    module: Any,
) -> dict[str, Any]:
    if not config.create_report:
        return {}

    function = _find_callable(
        module,
        ("generate_report", "create_report", "save_report"),
    )
    if function is None:
        message = "report.py enthält keine unterstützte Berichtsfunktion."
        if config.strict_modules:
            raise RuntimeError(message)
        LOGGER.warning(message)
        return {}

    result = _call_supported(
        function,
        output_dir=config.output_dir / "report",
        metrics=metrics,
        figures=figures,
        video_info=video_summary,
        imu_info=imu_summary,
        preprocessing_info=preprocessing_info,
        synchronization_info=synchronization_info,
        warnings=warnings,
        title=config.report_title,
        run_name=config.run_name,
        author=config.author,
        description=(
            "Automatisch erzeugter Vergleich eines Video-Tracking-Signals "
            "mit einer IMU-Referenz."
        ),
        extra={
            "configuration": dataclasses.asdict(config),
            "created_utc": _utc_now(),
            **config.extra,
        },
        overwrite=config.overwrite,
    )
    return _jsonable(result)


# =============================================================================
# Hauptpipeline
# =============================================================================

def validate_video_vs_imu(
    config: ValidationConfig,
) -> ValidationResult:
    """Führt die vollständige Validierung aus."""

    started = perf_counter()
    warnings: list[str] = []

    if not config.video_path.is_file():
        raise FileNotFoundError(
            f"Video-Tracking-Datei fehlt: {config.video_path}"
        )

    if not config.imu_path.is_file():
        raise FileNotFoundError(
            f"IMU-Datei fehlt: {config.imu_path}"
        )

    if (
        config.output_dir.exists()
        and not config.output_dir.is_dir()
    ):
        raise NotADirectoryError(config.output_dir)

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    module_paths = {
        "video_loader": "validation.video_loader",
        "imu_loader": "validation.imu_loader",
        "preprocessing": "validation.preprocessing",
        "synchronization": "validation.synchronization",
        "metrics": "validation.metrics",
        "plotting": "validation.plotting",
        "report": "validation.report",
        
    }

    modules = {
        key: _import_optional_module(module_path)
        for key, module_path in module_paths.items()
    }

    LOGGER.info(
        "1/7 Video-Tracking-Signal laden"
    )

    raw_video = load_video_signal(
        config,
        modules["video_loader"],
    )

    LOGGER.info(
        "2/7 IMU-Signal laden"
    )

    raw_imu = load_imu_signal(
        config,
        modules["imu_loader"],
    )

    LOGGER.info(
        "3/7 Signale vorverarbeiten"
    )

    processed_video = preprocess_signal(
        raw_video,
        kind="video",
        scale=config.video_scale,
        offset=config.video_offset,
        config=config,
        module=modules["preprocessing"],
    )

    processed_imu = preprocess_signal(
        raw_imu,
        kind="imu",
        scale=config.imu_scale,
        offset=config.imu_offset,
        config=config,
        module=modules["preprocessing"],
    )

    LOGGER.info(
        "4/7 Signale synchronisieren"
    )

    (
        aligned_video,
        aligned_imu,
        synchronization_info,
    ) = synchronize_signals(
        processed_video,
        processed_imu,
        config,
        modules["synchronization"],
    )

    LOGGER.info(
        "5/7 Kennzahlen berechnen"
    )

    metric_results = compute_metrics(
        aligned_video,
        aligned_imu,
        synchronization_info,
        config,
        modules["metrics"],
    )

    metric_results.setdefault(
        "estimated_lag_s",
        synchronization_info.get("lag_s"),
    )

    metric_results.setdefault(
        "correlation_at_lag",
        synchronization_info.get(
            "correlation_at_lag"
        ),
    )

    if config.save_intermediate:
        _save_intermediate(
            config.output_dir,
            raw_video,
            raw_imu,
            processed_video,
            processed_imu,
            aligned_video,
            aligned_imu,
        )

    LOGGER.info(
        "6/7 Abbildungen erzeugen"
    )

    figure_paths = create_plots(
        aligned_video,
        aligned_imu,
        metric_results,
        synchronization_info,
        config,
        modules["plotting"],
    )

    video_summary = raw_video.summary()
    imu_summary = raw_imu.summary()

    preprocessing_info = {
        "video": processed_video.metadata.get(
            "preprocessing",
            {},
        ),
        "imu": processed_imu.metadata.get(
            "preprocessing",
            {},
        ),
        "video_function": processed_video.metadata.get(
            "preprocessing_function",
            processed_video.metadata.get("loader"),
        ),
        "imu_function": processed_imu.metadata.get(
            "preprocessing_function",
            processed_imu.metadata.get("loader"),
        ),
    }

    LOGGER.info(
        "7/7 Bericht erzeugen"
    )

    report_artifacts = _generate_report(
        config,
        metric_results,
        figure_paths,
        video_summary,
        imu_summary,
        preprocessing_info,
        synchronization_info,
        warnings,
        modules["report"],
    )

    runtime = perf_counter() - started

    result = ValidationResult(
        output_dir=config.output_dir.resolve(),
        metrics=dict(
            _jsonable(metric_results)
        ),
        synchronization=dict(
            _jsonable(synchronization_info)
        ),
        video_summary=video_summary,
        imu_summary=imu_summary,
        figure_paths=[
            path.resolve()
            for path in figure_paths
        ],
        report_artifacts=report_artifacts,
        warnings=warnings,
        runtime_s=runtime,
    )

    _write_json(
        config.output_dir / "validation_result.json",
        result.as_dict(),
    )

    _write_json(
        config.output_dir / "validation_config.json",
        dataclasses.asdict(config),
    )

    LOGGER.info(
        "Validierung abgeschlossen: %.3f s, Ausgabe: %s",
        runtime,
        config.output_dir.resolve(),
    )

    return result

# =============================================================================
# Konfigurationsdatei und CLI
# =============================================================================

def _load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "Für YAML-Konfigurationen wird PyYAML benötigt."
                ) from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError("Konfiguration muss JSON oder YAML sein.")

    if not isinstance(data, Mapping):
        raise TypeError("Die Konfigurationsdatei muss ein Objekt enthalten.")
    return dict(data)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Validiert ein Video-Tracking-Signal gegen ein IMU-Signal.",
    )

    parser.add_argument("--config", type=Path)
    parser.add_argument("--video-csv", "--video-path", dest="video_path", type=Path)
    parser.add_argument("--imu-csv", "--imu-path", dest="imu_path", type=Path)
    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--video-time-column")
    parser.add_argument("--video-value-column")
    parser.add_argument("--imu-time-column")
    parser.add_argument("--imu-value-column")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--imu-sample-rate-hz", type=float)
    parser.add_argument("--target-sample-rate-hz", type=float)

    parser.add_argument("--video-scale", type=float)
    parser.add_argument("--video-offset", type=float)
    parser.add_argument("--imu-scale", type=float)
    parser.add_argument("--imu-offset", type=float)

    parser.add_argument("--start-time-s", type=float)
    parser.add_argument("--end-time-s", type=float)
    parser.add_argument("--smoothing-window", type=int)
    parser.add_argument("--lowpass-hz", type=float)
    parser.add_argument("--highpass-hz", type=float)
    parser.add_argument(
        "--normalize",
        choices=["none", "zscore", "minmax", "robust"],
    )
    parser.add_argument("--no-detrend", action="store_true", default=None)
    parser.add_argument("--no-center", action="store_true", default=None)

    parser.add_argument(
        "--synchronization-method",
        choices=["cross_correlation", "timestamp", "none"],
    )
    parser.add_argument("--max-lag-s", type=float)
    parser.add_argument("--manual-lag-s", type=float)

    parser.add_argument("--video-name")
    parser.add_argument("--imu-name")
    parser.add_argument("--video-unit")
    parser.add_argument("--imu-unit")
    parser.add_argument("--report-title")
    parser.add_argument("--run-name")
    parser.add_argument("--author")

    parser.add_argument("--no-intermediate", action="store_true", default=None)
    parser.add_argument("--no-plots", action="store_true", default=None)
    parser.add_argument("--no-report", action="store_true", default=None)
    parser.add_argument("--strict-modules", action="store_true", default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    return parser


def _merge_cli_and_file(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if args.config:
        data.update(_load_config_file(args.config))

    direct_names = (
        "video_path",
        "imu_path",
        "output_dir",
        "video_time_column",
        "video_value_column",
        "imu_time_column",
        "imu_value_column",
        "video_fps",
        "imu_sample_rate_hz",
        "target_sample_rate_hz",
        "video_scale",
        "video_offset",
        "imu_scale",
        "imu_offset",
        "start_time_s",
        "end_time_s",
        "smoothing_window",
        "lowpass_hz",
        "highpass_hz",
        "normalize",
        "synchronization_method",
        "max_lag_s",
        "manual_lag_s",
        "video_name",
        "imu_name",
        "video_unit",
        "imu_unit",
        "report_title",
        "run_name",
        "author",
    )
    for name in direct_names:
        value = getattr(args, name, None)
        if value is not None:
            data[name] = value

    boolean_mapping = {
        "no_detrend": ("detrend", False),
        "no_center": ("center", False),
        "no_intermediate": ("save_intermediate", False),
        "no_plots": ("create_plots", False),
        "no_report": ("create_report", False),
        "strict_modules": ("strict_modules", True),
    }
    for argument_name, (config_name, value) in boolean_mapping.items():
        if getattr(args, argument_name, None):
            data[config_name] = value

    aliases = {
        "video_csv": "video_path",
        "imu_csv": "imu_path",
    }
    for old, new in aliases.items():
        if old in data and new not in data:
            data[new] = data.pop(old)

    return data


def _build_config(data: MutableMapping[str, Any]) -> ValidationConfig:
    required = ("video_path", "imu_path", "output_dir")
    missing = [name for name in required if not data.get(name)]
    if missing:
        raise ValueError(
            "Fehlende Pflichtangaben: "
            + ", ".join(missing)
            + ". Diese können per CLI oder Konfigurationsdatei gesetzt werden."
        )

    valid_fields = {field.name for field in dataclasses.fields(ValidationConfig)}
    unknown = sorted(set(data) - valid_fields)
    extra = dict(data.get("extra", {}))
    for key in unknown:
        extra[key] = data.pop(key)
    data["extra"] = extra

    return ValidationConfig(**data)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        config_data = _merge_cli_and_file(args)
        config = _build_config(config_data)
        result = validate_video_vs_imu(config)
    except KeyboardInterrupt:
        LOGGER.error("Validierung wurde abgebrochen.")
        return 130
    except Exception as exc:
        LOGGER.error("Validierung fehlgeschlagen: %s", exc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        return 1

    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())