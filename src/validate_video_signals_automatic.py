#!/usr/bin/env python3
"""
Automatische Stage-3-Validierung mehrerer Video-Signalspalten.

Das Skript startet `validate_top_candidates_professional.py` für jede angegebene
Video-Spalte, liest anschließend die jeweilige `comparison.csv` ein und erzeugt
eine gemeinsame Rangliste.

Beispiel:
    python src/validate_video_signals_automatic.py \
      --video-csv outputs/Lucas_Kanade_CPU7/GX010049_results_inner_pipe_track_timeseries.csv \
      --candidates-csv outputs/candidate_search/validation_candidates_ranked.csv \
      --output-dir outputs/signal_validation/GX010049 \
      --video-id GX010049 \
      --top-k 5

Standardmäßig werden diese Spalten untersucht:
    inner_pipe_track_center_x
    inner_pipe_track_center_y
    inner_pipe_track_area
    inner_pipe_seg_center_x
    inner_pipe_seg_center_y
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "pandas ist erforderlich. Aktiviere zuerst deine virtuelle Umgebung."
    ) from exc


LOGGER = logging.getLogger("validate_video_signals_automatic")

DEFAULT_SIGNALS = (
    "inner_pipe_track_center_x",
    "inner_pipe_track_center_y",
    "inner_pipe_track_area",
    "inner_pipe_seg_center_x",
    "inner_pipe_seg_center_y",
)


@dataclass
class SignalResult:
    signal: str
    status: str
    return_code: int
    runtime_s: float
    output_dir: str
    comparison_csv: str
    best_final_rank: Optional[int] = None
    best_source_rank: Optional[int] = None
    best_rank_for_video: Optional[int] = None
    best_label: Optional[str] = None
    best_imu_measurement_id: Optional[str] = None
    best_imu_path: Optional[str] = None
    pearson_correlation: Optional[float] = None
    absolute_pearson_correlation: Optional[float] = None
    rmse: Optional[float] = None
    normalized_rmse: Optional[float] = None
    mae: Optional[float] = None
    r_squared: Optional[float] = None
    estimated_lag_s: Optional[float] = None
    correlation_at_lag: Optional[float] = None
    aligned_samples: Optional[int] = None
    sample_rate_hz: Optional[float] = None
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Führt Stage 3 automatisch für mehrere Video-Signalspalten aus "
            "und erstellt eine gemeinsame Ergebnisrangliste."
        )
    )

    parser.add_argument("--video-csv", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-id", required=True)

    parser.add_argument(
        "--validator-script",
        type=Path,
        default=Path(__file__).resolve().parent
        / "validate_top_candidates_professional.py",
        help="Pfad zu validate_top_candidates_professional.py",
    )
    parser.add_argument(
        "--video-time-column",
        default="time_seconds",
        help="Zeitspalte des Video-CSV.",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=list(DEFAULT_SIGNALS),
        help="Zu testende Video-Signalspalten.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--ranking-mode",
        choices=("validator", "absolute_pearson", "pearson", "rmse"),
        default="absolute_pearson",
        help=(
            "Globale Rangfolge der Videosignale. "
            "'validator' nutzt die beste Zeile aus comparison.csv; "
            "'absolute_pearson' bevorzugt den größten Betrag der Korrelation."
        ),
    )
    parser.add_argument(
        "--candidate-selection",
        choices=("final_rank", "absolute_pearson", "pearson", "rmse"),
        default="final_rank",
        help="Auswahl des besten IMU-Kandidaten innerhalb eines Signallaufs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vorhandene Signalläufe erneut ausführen.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Nach einem fehlgeschlagenen Signallauf fortfahren.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Beim ersten fehlgeschlagenen Signallauf abbrechen.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optionales Zeitlimit pro Signallauf in Sekunden.",
    )
    parser.add_argument(
        "--extra-validator-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Zusätzliche Argumente für validate_top_candidates_professional.py. "
            "Diese Option muss am Ende des Befehls stehen."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    return parser.parse_args()


def configure_logging(level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "automatic_signal_validation.log"

    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "signal"


def validate_inputs(args: argparse.Namespace) -> None:
    missing = [
        path
        for path in (
            args.video_csv,
            args.candidates_csv,
            args.validator_script,
        )
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Erforderliche Dateien fehlen:\n{formatted}")

    if args.top_k <= 0:
        raise ValueError("--top-k muss größer als 0 sein.")

    frame = pd.read_csv(args.video_csv, nrows=5)
    required = [args.video_time_column, *args.signals]
    missing_columns = [column for column in required if column not in frame.columns]

    if missing_columns:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(
            "Folgende Spalten fehlen im Video-CSV: "
            f"{missing_columns}\nVerfügbare Spalten: {available}"
        )


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_int(value: Any) -> Optional[int]:
    number = finite_float(value)
    return int(number) if number is not None else None


def text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def choose_best_candidate(
    frame: pd.DataFrame,
    selection: str,
) -> pd.Series:
    successful = frame.copy()

    if "status" in successful.columns:
        successful = successful[
            successful["status"].astype(str).str.lower().eq("success")
        ]

    if successful.empty:
        raise ValueError("comparison.csv enthält keinen erfolgreichen Kandidaten.")

    if selection == "final_rank":
        if "final_rank" not in successful.columns:
            raise ValueError("Spalte 'final_rank' fehlt in comparison.csv.")
        return successful.sort_values(
            "final_rank", ascending=True, na_position="last"
        ).iloc[0]

    if selection in {"absolute_pearson", "pearson"}:
        column = "pearson_correlation"
        if column not in successful.columns:
            raise ValueError(f"Spalte '{column}' fehlt in comparison.csv.")
        numeric = pd.to_numeric(successful[column], errors="coerce")
        successful = successful.assign(_metric=numeric)
        successful = successful[successful["_metric"].notna()]
        if successful.empty:
            raise ValueError("Keine gültigen Pearson-Werte vorhanden.")
        if selection == "absolute_pearson":
            successful = successful.assign(_metric=successful["_metric"].abs())
        return successful.sort_values("_metric", ascending=False).iloc[0]

    if selection == "rmse":
        if "rmse" not in successful.columns:
            raise ValueError("Spalte 'rmse' fehlt in comparison.csv.")
        numeric = pd.to_numeric(successful["rmse"], errors="coerce")
        successful = successful.assign(_metric=numeric)
        successful = successful[successful["_metric"].notna()]
        if successful.empty:
            raise ValueError("Keine gültigen RMSE-Werte vorhanden.")
        return successful.sort_values("_metric", ascending=True).iloc[0]

    raise ValueError(f"Unbekannte Auswahlmethode: {selection}")


def build_result_from_row(
    signal: str,
    output_dir: Path,
    comparison_csv: Path,
    runtime_s: float,
    return_code: int,
    row: pd.Series,
) -> SignalResult:
    pearson = finite_float(row.get("pearson_correlation"))

    return SignalResult(
        signal=signal,
        status="success",
        return_code=return_code,
        runtime_s=runtime_s,
        output_dir=str(output_dir.resolve()),
        comparison_csv=str(comparison_csv.resolve()),
        best_final_rank=finite_int(row.get("final_rank")),
        best_source_rank=finite_int(row.get("source_rank")),
        best_rank_for_video=finite_int(row.get("rank_for_video")),
        best_label=text_or_none(row.get("label")),
        best_imu_measurement_id=text_or_none(row.get("imu_measurement_id")),
        best_imu_path=text_or_none(row.get("imu_path")),
        pearson_correlation=pearson,
        absolute_pearson_correlation=(
            abs(pearson) if pearson is not None else None
        ),
        rmse=finite_float(row.get("rmse")),
        normalized_rmse=finite_float(row.get("normalized_rmse")),
        mae=finite_float(row.get("mae")),
        r_squared=finite_float(row.get("r_squared")),
        estimated_lag_s=finite_float(row.get("estimated_lag_s")),
        correlation_at_lag=finite_float(row.get("correlation_at_lag")),
        aligned_samples=finite_int(row.get("aligned_samples")),
        sample_rate_hz=finite_float(row.get("sample_rate_hz")),
    )


def existing_successful_result(
    signal: str,
    signal_output_dir: Path,
    selection: str,
) -> Optional[SignalResult]:
    comparison_csv = signal_output_dir / "comparison.csv"
    if not comparison_csv.is_file():
        return None

    try:
        frame = pd.read_csv(comparison_csv)
        row = choose_best_candidate(frame, selection)
        return build_result_from_row(
            signal=signal,
            output_dir=signal_output_dir,
            comparison_csv=comparison_csv,
            runtime_s=0.0,
            return_code=0,
            row=row,
        )
    except Exception as exc:
        LOGGER.warning(
            "Vorhandener Lauf für %s ist nicht wiederverwendbar: %s",
            signal,
            exc,
        )
        return None


def run_signal(
    args: argparse.Namespace,
    signal: str,
    index: int,
    total: int,
) -> SignalResult:
    signal_dir = args.output_dir / sanitize_name(signal)
    comparison_csv = signal_dir / "comparison.csv"

    if not args.force:
        existing = existing_successful_result(
            signal,
            signal_dir,
            args.candidate_selection,
        )
        if existing is not None:
            existing.status = "resumed"
            LOGGER.info(
                "[%d/%d] %s: vorhandenes Ergebnis wird verwendet.",
                index,
                total,
                signal,
            )
            return existing

    signal_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = signal_dir / "validator_stdout.log"

    command = [
        sys.executable,
        str(args.validator_script.resolve()),
        "--video-csv",
        str(args.video_csv.resolve()),
        "--candidates-csv",
        str(args.candidates_csv.resolve()),
        "--output-dir",
        str(signal_dir.resolve()),
        "--video-id",
        str(args.video_id),
        "--video-time-column",
        str(args.video_time_column),
        "--video-value-column",
        signal,
        "--top-k",
        str(args.top_k),
        *args.extra_validator_args,
    ]

    LOGGER.info(
        "[%d/%d] Starte Signal: %s",
        index,
        total,
        signal,
    )
    LOGGER.debug("Befehl: %s", " ".join(command))

    started = time.perf_counter()

    try:
        with stdout_path.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        runtime_s = time.perf_counter() - started

    except subprocess.TimeoutExpired:
        runtime_s = time.perf_counter() - started
        return SignalResult(
            signal=signal,
            status="timeout",
            return_code=124,
            runtime_s=runtime_s,
            output_dir=str(signal_dir.resolve()),
            comparison_csv=str(comparison_csv.resolve()),
            error=f"Zeitlimit von {args.timeout} Sekunden überschritten.",
        )

    if process.returncode != 0:
        return SignalResult(
            signal=signal,
            status="failure",
            return_code=process.returncode,
            runtime_s=runtime_s,
            output_dir=str(signal_dir.resolve()),
            comparison_csv=str(comparison_csv.resolve()),
            error=(
                "Validator wurde mit Fehlercode "
                f"{process.returncode} beendet. Siehe {stdout_path}."
            ),
        )

    if not comparison_csv.is_file():
        return SignalResult(
            signal=signal,
            status="failure",
            return_code=process.returncode,
            runtime_s=runtime_s,
            output_dir=str(signal_dir.resolve()),
            comparison_csv=str(comparison_csv.resolve()),
            error="comparison.csv wurde nicht erzeugt.",
        )

    try:
        frame = pd.read_csv(comparison_csv)
        row = choose_best_candidate(frame, args.candidate_selection)
        result = build_result_from_row(
            signal=signal,
            output_dir=signal_dir,
            comparison_csv=comparison_csv,
            runtime_s=runtime_s,
            return_code=process.returncode,
            row=row,
        )
        LOGGER.info(
            "[%d/%d] %s abgeschlossen: Pearson=%s, RMSE=%s, IMU=%s",
            index,
            total,
            signal,
            result.pearson_correlation,
            result.rmse,
            result.best_label or result.best_imu_measurement_id,
        )
        return result

    except Exception as exc:
        return SignalResult(
            signal=signal,
            status="failure",
            return_code=process.returncode,
            runtime_s=runtime_s,
            output_dir=str(signal_dir.resolve()),
            comparison_csv=str(comparison_csv.resolve()),
            error=f"comparison.csv konnte nicht ausgewertet werden: {exc}",
        )


def sort_results(
    results: Sequence[SignalResult],
    ranking_mode: str,
) -> list[SignalResult]:
    successful = [
        result
        for result in results
        if result.status in {"success", "resumed"}
    ]
    failed = [
        result
        for result in results
        if result.status not in {"success", "resumed"}
    ]

    def numeric_or_default(
        value: Optional[float],
        default: float,
    ) -> float:
        return value if value is not None else default

    if ranking_mode == "absolute_pearson":
        successful.sort(
            key=lambda item: (
                -numeric_or_default(
                    item.absolute_pearson_correlation,
                    -math.inf,
                ),
                numeric_or_default(item.rmse, math.inf),
            )
        )
    elif ranking_mode == "pearson":
        successful.sort(
            key=lambda item: (
                -numeric_or_default(item.pearson_correlation, -math.inf),
                numeric_or_default(item.rmse, math.inf),
            )
        )
    elif ranking_mode == "rmse":
        successful.sort(
            key=lambda item: (
                numeric_or_default(item.rmse, math.inf),
                -numeric_or_default(
                    item.absolute_pearson_correlation,
                    -math.inf,
                ),
            )
        )
    elif ranking_mode == "validator":
        successful.sort(
            key=lambda item: (
                numeric_or_default(item.best_final_rank, math.inf),
                -numeric_or_default(
                    item.absolute_pearson_correlation,
                    -math.inf,
                ),
            )
        )

    return [*successful, *failed]


def write_csv(path: Path, results: Sequence[SignalResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    args: argparse.Namespace,
    results: Sequence[SignalResult],
) -> None:
    payload = {
        "video_id": args.video_id,
        "video_csv": str(args.video_csv.resolve()),
        "candidates_csv": str(args.candidates_csv.resolve()),
        "video_time_column": args.video_time_column,
        "signals": list(args.signals),
        "top_k": args.top_k,
        "ranking_mode": args.ranking_mode,
        "candidate_selection": args.candidate_selection,
        "results": [asdict(result) for result in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_number(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    results: Sequence[SignalResult],
) -> None:
    successful = [
        result
        for result in results
        if result.status in {"success", "resumed"}
    ]

    lines = [
        f"# Automatische Videosignal-Validierung – {args.video_id}",
        "",
        f"- Video-CSV: `{args.video_csv.resolve()}`",
        f"- Kandidaten-CSV: `{args.candidates_csv.resolve()}`",
        f"- Zeitspalte: `{args.video_time_column}`",
        f"- Top-K IMU-Kandidaten: `{args.top_k}`",
        f"- Rangmodus: `{args.ranking_mode}`",
        "",
        "| Rang | Videosignal | Pearson | |Pearson| | RMSE | Lag [s] | Beste IMU-Messung | Status |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]

    for rank, result in enumerate(successful, start=1):
        label = result.best_label or result.best_imu_measurement_id or ""
        lines.append(
            "| "
            f"{rank} | `{result.signal}` | "
            f"{format_number(result.pearson_correlation)} | "
            f"{format_number(result.absolute_pearson_correlation)} | "
            f"{format_number(result.rmse)} | "
            f"{format_number(result.estimated_lag_s)} | "
            f"{label} | {result.status} |"
        )

    failures = [
        result
        for result in results
        if result.status not in {"success", "resumed"}
    ]
    if failures:
        lines.extend(["", "## Fehlgeschlagene Signale", ""])
        for result in failures:
            lines.append(
                f"- `{result.signal}`: {result.error or result.status}"
            )

    if successful:
        best = successful[0]
        lines.extend(
            [
                "",
                "## Bestes Videosignal",
                "",
                f"`{best.signal}`",
                "",
                f"- Pearson-Korrelation: {format_number(best.pearson_correlation)}",
                f"- Betrag der Pearson-Korrelation: {format_number(best.absolute_pearson_correlation)}",
                f"- RMSE: {format_number(best.rmse)}",
                f"- geschätzter Zeitversatz: {format_number(best.estimated_lag_s)} s",
                f"- beste IMU-Messung: {best.best_label or best.best_imu_measurement_id or ''}",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(results: Sequence[SignalResult]) -> None:
    successful = [
        result
        for result in results
        if result.status in {"success", "resumed"}
    ]

    print("=" * 100)
    print("AUTOMATISCHE VIDEOSIGNAL-VALIDIERUNG")
    print("=" * 100)

    if not successful:
        print("Kein Videosignal wurde erfolgreich validiert.")
    else:
        for rank, result in enumerate(successful, start=1):
            label = result.best_label or result.best_imu_measurement_id or "-"
            print(
                f"{rank:>2}. {result.signal:<32} "
                f"Pearson={format_number(result.pearson_correlation):>10} "
                f"|Pearson|={format_number(result.absolute_pearson_correlation):>10} "
                f"RMSE={format_number(result.rmse):>10} "
                f"Lag={format_number(result.estimated_lag_s):>10} "
                f"IMU={label}"
            )

    failed_count = len(results) - len(successful)
    print("-" * 100)
    print(
        f"Erfolgreich: {len(successful)} | Fehlgeschlagen: {failed_count}"
    )


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    configure_logging(args.log_level, args.output_dir)

    try:
        validate_inputs(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2

    results: list[SignalResult] = []
    total = len(args.signals)

    for index, signal in enumerate(args.signals, start=1):
        result = run_signal(args, signal, index, total)
        results.append(result)

        if result.status not in {"success", "resumed"}:
            LOGGER.error("%s fehlgeschlagen: %s", signal, result.error)
            if args.stop_on_error:
                break

    ranked = sort_results(results, args.ranking_mode)

    csv_path = args.output_dir / "signal_comparison.csv"
    json_path = args.output_dir / "signal_comparison.json"
    markdown_path = args.output_dir / "signal_comparison.md"

    write_csv(csv_path, ranked)
    write_json(json_path, args, ranked)
    write_markdown(markdown_path, args, ranked)
    print_summary(ranked)

    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {markdown_path}")

    successful = [
        result
        for result in ranked
        if result.status in {"success", "resumed"}
    ]
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())