#!/usr/bin/env python3
"""
Erzeugung einer robusten Lucas-Kanade-Zeitreihe für die Validierung
gegen IMU-Messdaten.

Die Eingabedatei enthält pro Frame mehrere unabhängig verfolgte
Lucas-Kanade-Punkte. Da die Punkte an unterschiedlichen absoluten
Bildpositionen liegen, dürfen ihre absoluten x- und y-Koordinaten
nicht direkt gemeinsam aggregiert werden.

Stattdessen wird:

1. für jeden point_id eine individuelle Referenzposition bestimmt,
2. für jeden Punkt die relative Verschiebung berechnet,
3. anschließend pro Frame der Median der relativen Verschiebungen
   aller gültigen Punkte gebildet.

Dadurch beeinflusst ein wechselnder Satz gültiger Punkte nicht mehr
direkt das resultierende Bewegungssignal.

Beispiel
--------
python -m src.create_lk_validation_timeseries

Oder mit expliziten Pfaden:

python -m src.create_lk_validation_timeseries \
    --input-csv /pfad/GX010262_lucas_kanade.csv \
    --output-csv /pfad/GX010262_lucas_kanade_timeseries.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


# ============================================================
# Repository-spezifische Standardpfade
# ============================================================

DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/Lucas_Kanade_CPU_1"
)

DEFAULT_INPUT_CSV: Final[Path] = (
    DEFAULT_OUTPUT_DIRECTORY / "GX010262_lucas_kanade.csv"
)

DEFAULT_OUTPUT_CSV: Final[Path] = (
    DEFAULT_OUTPUT_DIRECTORY / "GX010262_lucas_kanade_timeseries.csv"
)

DEFAULT_SUMMARY_JSON: Final[Path] = (
    DEFAULT_OUTPUT_DIRECTORY
    / "GX010262_lucas_kanade_timeseries_summary.json"
)


# ============================================================
# Qualitätsparameter
# ============================================================

DEFAULT_MAX_FB_ERROR: Final[float] = 1.0
DEFAULT_MAX_JUMP_PX: Final[float] = 20.0

# Mindestanzahl gleichzeitig gültiger Punkte, damit ein Frame
# für das aggregierte Validierungssignal verwendet wird.
DEFAULT_MIN_VALID_POINTS: Final[int] = 2

# Die Referenzposition eines Punktes wird robust aus den ersten
# gültigen Messwerten gebildet und nicht nur aus einem Einzelwert.
DEFAULT_REFERENCE_SAMPLES: Final[int] = 20


REQUIRED_COLUMNS: Final[set[str]] = {
    "frame",
    "time_seconds",
    "point_id",
    "x",
    "y",
    "dx",
    "dy",
    "jump_px",
    "fb_error",
    "tracking_status",
}

NUMERIC_COLUMNS: Final[list[str]] = [
    "frame",
    "time_seconds",
    "point_id",
    "x",
    "y",
    "dx",
    "dy",
    "jump_px",
    "fb_error",
    "tracking_status",
]


# ============================================================
# Argumente
# ============================================================

def parse_args() -> argparse.Namespace:
    """Liest die Kommandozeilenargumente ein."""

    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt aus einer punktweisen Lucas-Kanade-CSV eine "
            "robuste Zeitreihe für die Video-IMU-Validierung."
        )
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=(
            "Eingabe-CSV aus src.optical_flow_test. "
            f"Standard: {DEFAULT_INPUT_CSV}"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=(
            "Ausgabedatei der aggregierten LK-Zeitreihe. "
            f"Standard: {DEFAULT_OUTPUT_CSV}"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=(
            "Optionale JSON-Zusammenfassung. "
            f"Standard: {DEFAULT_SUMMARY_JSON}"
        ),
    )

    parser.add_argument(
        "--max-fb-error",
        type=float,
        default=DEFAULT_MAX_FB_ERROR,
        help=(
            "Maximal erlaubter Forward-Backward-Fehler. "
            f"Standard: {DEFAULT_MAX_FB_ERROR}"
        ),
    )

    parser.add_argument(
        "--max-jump-px",
        type=float,
        default=DEFAULT_MAX_JUMP_PX,
        help=(
            "Maximal erlaubte Punktbewegung zwischen zwei Frames. "
            f"Standard: {DEFAULT_MAX_JUMP_PX}"
        ),
    )

    parser.add_argument(
        "--min-valid-points",
        type=int,
        default=DEFAULT_MIN_VALID_POINTS,
        help=(
            "Mindestanzahl gültiger Punkte für einen aggregierten "
            f"Frame. Standard: {DEFAULT_MIN_VALID_POINTS}"
        ),
    )

    parser.add_argument(
        "--reference-samples",
        type=int,
        default=DEFAULT_REFERENCE_SAMPLES,
        help=(
            "Anzahl der ersten gültigen Werte pro Punkt, aus denen "
            "die Referenzposition als Median bestimmt wird. "
            f"Standard: {DEFAULT_REFERENCE_SAMPLES}"
        ),
    )

    parser.add_argument(
        "--point-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional: Nur bestimmte Punkt-IDs verwenden, "
            "beispielsweise --point-ids 0 1 2 3 4. "
            "Ohne Angabe werden alle Punkte verwendet."
        ),
    )

    return parser.parse_args()


# ============================================================
# Einlesen und Validieren
# ============================================================

def load_input_csv(path: Path) -> pd.DataFrame:
    """
    Liest die Lucas-Kanade-CSV ein und prüft die benötigten Spalten.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Lucas-Kanade-Eingabedatei nicht gefunden: {path}"
        )

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"CSV konnte nicht gelesen werden: {path}"
        ) from exc

    if dataframe.empty:
        raise ValueError(f"Die Eingabedatei ist leer: {path}")

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Die Lucas-Kanade-CSV enthält nicht alle benötigten "
            "Spalten.\n"
            f"Fehlende Spalten: {sorted(missing_columns)}\n"
            f"Vorhandene Spalten: {list(dataframe.columns)}"
        )

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "frame",
            "time_seconds",
            "point_id",
            "x",
            "y",
            "tracking_status",
        ]
    ).copy()

    if dataframe.empty:
        raise RuntimeError(
            "Nach der numerischen Konvertierung sind keine "
            "verwertbaren Lucas-Kanade-Zeilen vorhanden."
        )

    dataframe["frame"] = dataframe["frame"].astype(int)
    dataframe["point_id"] = dataframe["point_id"].astype(int)
    dataframe["tracking_status"] = (
        dataframe["tracking_status"].astype(int)
    )

    dataframe = dataframe.sort_values(
        ["frame", "point_id"]
    ).reset_index(drop=True)

    return dataframe


def validate_parameters(args: argparse.Namespace) -> None:
    """Prüft die numerischen Kommandozeilenparameter."""

    if not np.isfinite(args.max_fb_error):
        raise ValueError("--max-fb-error muss endlich sein.")

    if args.max_fb_error < 0:
        raise ValueError("--max-fb-error darf nicht negativ sein.")

    if not np.isfinite(args.max_jump_px):
        raise ValueError("--max-jump-px muss endlich sein.")

    if args.max_jump_px <= 0:
        raise ValueError("--max-jump-px muss größer als 0 sein.")

    if args.min_valid_points < 1:
        raise ValueError(
            "--min-valid-points muss mindestens 1 sein."
        )

    if args.reference_samples < 1:
        raise ValueError(
            "--reference-samples muss mindestens 1 sein."
        )


# ============================================================
# Qualitätsfilter
# ============================================================

def filter_valid_measurements(
    dataframe: pd.DataFrame,
    *,
    max_fb_error: float,
    max_jump_px: float,
    selected_point_ids: list[int] | None,
) -> pd.DataFrame:
    """
    Wendet die Qualitätskriterien auf die LK-Messungen an.

    Eine Messung gilt nur dann als gültig, wenn:

    - tracking_status == 1,
    - x und y endlich sind,
    - fb_error endlich und unterhalb des Grenzwerts liegt,
    - jump_px endlich und unterhalb des Grenzwerts liegt.
    """

    working = dataframe.copy()

    if selected_point_ids is not None:
        selected_set = set(selected_point_ids)

        working = working[
            working["point_id"].isin(selected_set)
        ].copy()

        if working.empty:
            available_ids = sorted(
                dataframe["point_id"].unique().tolist()
            )

            raise RuntimeError(
                "Für die gewählten Punkt-IDs wurden keine Daten "
                "gefunden.\n"
                f"Gewählt: {sorted(selected_set)}\n"
                f"Vorhanden: {available_ids}"
            )

    valid_mask = (
        (working["tracking_status"] == 1)
        & np.isfinite(working["x"])
        & np.isfinite(working["y"])
        & np.isfinite(working["fb_error"])
        & np.isfinite(working["jump_px"])
        & (working["fb_error"] <= max_fb_error)
        & (working["jump_px"] <= max_jump_px)
    )

    valid = working.loc[valid_mask].copy()

    if valid.empty:
        raise RuntimeError(
            "Nach der Qualitätsfilterung sind keine gültigen "
            "Lucas-Kanade-Messungen übrig."
        )

    return valid


# ============================================================
# Punktweise Referenzposition
# ============================================================

def calculate_point_references(
    valid: pd.DataFrame,
    *,
    reference_samples: int,
) -> pd.DataFrame:
    """
    Bestimmt für jeden Punkt eine individuelle Referenzposition.

    Verwendet wird der Median der ersten N gültigen Messwerte des
    jeweiligen Punktes. Das ist robuster als ein einzelner erster
    Messwert.
    """

    reference_rows: list[dict[str, float | int]] = []

    for point_id, point_data in valid.groupby(
        "point_id",
        sort=True,
    ):
        point_data = point_data.sort_values(
            ["time_seconds", "frame"]
        )

        reference_data = point_data.head(reference_samples)

        if reference_data.empty:
            continue

        x_reference = float(reference_data["x"].median())
        y_reference = float(reference_data["y"].median())

        if not np.isfinite(x_reference):
            continue

        if not np.isfinite(y_reference):
            continue

        reference_rows.append(
            {
                "point_id": int(point_id),
                "reference_x": x_reference,
                "reference_y": y_reference,
                "reference_sample_count": int(
                    len(reference_data)
                ),
                "reference_first_frame": int(
                    reference_data["frame"].min()
                ),
                "reference_last_frame": int(
                    reference_data["frame"].max()
                ),
            }
        )

    references = pd.DataFrame(reference_rows)

    if references.empty:
        raise RuntimeError(
            "Für keinen Lucas-Kanade-Punkt konnte eine gültige "
            "Referenzposition bestimmt werden."
        )

    return references


def add_pointwise_displacements(
    valid: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fügt relative Verschiebungen für jeden einzelnen Punkt hinzu.
    """

    pointwise = valid.merge(
        references,
        on="point_id",
        how="inner",
        validate="many_to_one",
    )

    if pointwise.empty:
        raise RuntimeError(
            "Die gültigen LK-Messungen konnten nicht mit den "
            "Punktreferenzen verbunden werden."
        )

    pointwise["point_displacement_x"] = (
        pointwise["x"] - pointwise["reference_x"]
    )

    pointwise["point_displacement_y"] = (
        pointwise["y"] - pointwise["reference_y"]
    )

    pointwise["point_displacement_magnitude"] = np.hypot(
        pointwise["point_displacement_x"],
        pointwise["point_displacement_y"],
    )

    return pointwise


# ============================================================
# Frameweise Aggregation
# ============================================================

def aggregate_frames(
    pointwise: pd.DataFrame,
    *,
    min_valid_points: int,
) -> pd.DataFrame:
    """
    Aggregiert die punktweisen Verschiebungen robust pro Frame.

    Entscheidend ist, dass hier relative Punktverschiebungen und
    nicht die absoluten Punktpositionen aggregiert werden.
    """

    timeseries = (
        pointwise.groupby(
            ["frame", "time_seconds"],
            as_index=False,
        )
        .agg(
            # Median der absoluten Koordinaten nur zu
            # Diagnosezwecken. Diese Spalten sind nicht das
            # primäre Validierungssignal.
            lk_center_x=("x", "median"),
            lk_center_y=("y", "median"),

            # Primäres Bewegungssignal für die Validierung.
            lk_displacement_x=(
                "point_displacement_x",
                "median",
            ),
            lk_displacement_y=(
                "point_displacement_y",
                "median",
            ),

            # Frame-zu-Frame-LK-Bewegung.
            lk_median_dx=("dx", "median"),
            lk_median_dy=("dy", "median"),

            # Qualitätsmetriken.
            lk_median_jump_px=("jump_px", "median"),
            lk_max_jump_px=("jump_px", "max"),
            lk_median_fb_error=("fb_error", "median"),
            lk_max_fb_error=("fb_error", "max"),
            valid_point_count=("point_id", "nunique"),

            # Streuung zwischen den Punkten.
            lk_displacement_x_std=(
                "point_displacement_x",
                "std",
            ),
            lk_displacement_y_std=(
                "point_displacement_y",
                "std",
            ),
            lk_displacement_x_min=(
                "point_displacement_x",
                "min",
            ),
            lk_displacement_x_max=(
                "point_displacement_x",
                "max",
            ),
            lk_displacement_y_min=(
                "point_displacement_y",
                "min",
            ),
            lk_displacement_y_max=(
                "point_displacement_y",
                "max",
            ),
        )
        .sort_values(["time_seconds", "frame"])
        .reset_index(drop=True)
    )

    timeseries = timeseries[
        timeseries["valid_point_count"] >= min_valid_points
    ].copy()

    if timeseries.empty:
        raise RuntimeError(
            "Kein Frame erfüllt die Mindestanzahl gültiger "
            f"Punkte von {min_valid_points}."
        )

    timeseries["lk_displacement_magnitude"] = np.hypot(
        timeseries["lk_displacement_x"],
        timeseries["lk_displacement_y"],
    )

    timeseries["lk_displacement_x_range"] = (
        timeseries["lk_displacement_x_max"]
        - timeseries["lk_displacement_x_min"]
    )

    timeseries["lk_displacement_y_range"] = (
        timeseries["lk_displacement_y_max"]
        - timeseries["lk_displacement_y_min"]
    )

    return timeseries


# ============================================================
# Punktweise Diagnosespalten
# ============================================================

def create_point_diagnostic_columns(
    pointwise: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine breite Tabelle mit Verschiebungen pro point_id.

    Beispiele:

    - point_0_displacement_x
    - point_0_displacement_y
    - point_0_fb_error
    - point_0_jump_px
    """

    diagnostic_source = pointwise[
        [
            "frame",
            "time_seconds",
            "point_id",
            "point_displacement_x",
            "point_displacement_y",
            "fb_error",
            "jump_px",
        ]
    ].copy()

    value_columns = [
        "point_displacement_x",
        "point_displacement_y",
        "fb_error",
        "jump_px",
    ]

    wide = diagnostic_source.pivot_table(
        index=["frame", "time_seconds"],
        columns="point_id",
        values=value_columns,
        aggfunc="first",
    )

    if wide.empty:
        return pd.DataFrame(
            columns=["frame", "time_seconds"]
        )

    renamed_columns: list[str] = []

    for measurement, point_id in wide.columns:
        if measurement == "point_displacement_x":
            suffix = "displacement_x"
        elif measurement == "point_displacement_y":
            suffix = "displacement_y"
        else:
            suffix = measurement

        renamed_columns.append(
            f"point_{int(point_id)}_{suffix}"
        )

    wide.columns = renamed_columns

    wide = wide.reset_index()

    return wide


def merge_diagnostics(
    timeseries: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Verbindet das Aggregat mit den Punktdiagnosespalten."""

    if diagnostics.empty:
        return timeseries

    merged = timeseries.merge(
        diagnostics,
        on=["frame", "time_seconds"],
        how="left",
        validate="one_to_one",
    )

    return merged.sort_values(
        ["time_seconds", "frame"]
    ).reset_index(drop=True)


# ============================================================
# Zusammenfassung
# ============================================================

def calculate_point_statistics(
    dataframe: pd.DataFrame,
    valid: pd.DataFrame,
    references: pd.DataFrame,
) -> list[dict[str, object]]:
    """Erzeugt Qualitätsstatistiken für jede point_id."""

    statistics: list[dict[str, object]] = []

    reference_lookup = references.set_index("point_id")

    point_ids = sorted(
        dataframe["point_id"].dropna().astype(int).unique()
    )

    for point_id in point_ids:
        raw_point = dataframe[
            dataframe["point_id"] == point_id
        ]

        valid_point = valid[
            valid["point_id"] == point_id
        ]

        total_count = int(len(raw_point))
        valid_count = int(len(valid_point))

        valid_fraction = (
            valid_count / total_count
            if total_count > 0
            else float("nan")
        )

        record: dict[str, object] = {
            "point_id": int(point_id),
            "total_measurement_count": total_count,
            "valid_measurement_count": valid_count,
            "valid_fraction": float(valid_fraction),
        }

        if not valid_point.empty:
            record.update(
                {
                    "median_fb_error": float(
                        valid_point["fb_error"].median()
                    ),
                    "max_fb_error": float(
                        valid_point["fb_error"].max()
                    ),
                    "median_jump_px": float(
                        valid_point["jump_px"].median()
                    ),
                    "max_jump_px": float(
                        valid_point["jump_px"].max()
                    ),
                    "first_valid_frame": int(
                        valid_point["frame"].min()
                    ),
                    "last_valid_frame": int(
                        valid_point["frame"].max()
                    ),
                }
            )

        if point_id in reference_lookup.index:
            reference_row = reference_lookup.loc[point_id]

            record.update(
                {
                    "reference_x": float(
                        reference_row["reference_x"]
                    ),
                    "reference_y": float(
                        reference_row["reference_y"]
                    ),
                    "reference_sample_count": int(
                        reference_row[
                            "reference_sample_count"
                        ]
                    ),
                }
            )

        statistics.append(record)

    return statistics


def build_summary(
    *,
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    valid: pd.DataFrame,
    references: pd.DataFrame,
    timeseries: pd.DataFrame,
) -> dict[str, object]:
    """Erzeugt eine maschinenlesbare Zusammenfassung."""

    total_frames = int(dataframe["frame"].nunique())
    valid_measurement_frames = int(valid["frame"].nunique())
    retained_frames = int(timeseries["frame"].nunique())

    if len(timeseries) >= 2:
        frame_time_differences = (
            timeseries["time_seconds"].diff().dropna()
        )

        median_time_step = float(
            frame_time_differences.median()
        )

        estimated_fps = (
            1.0 / median_time_step
            if median_time_step > 0
            else None
        )
    else:
        median_time_step = None
        estimated_fps = None

    return {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "summary_json": str(args.summary_json),
        "parameters": {
            "max_fb_error": float(args.max_fb_error),
            "max_jump_px": float(args.max_jump_px),
            "min_valid_points": int(
                args.min_valid_points
            ),
            "reference_samples": int(
                args.reference_samples
            ),
            "selected_point_ids": args.point_ids,
        },
        "input": {
            "row_count": int(len(dataframe)),
            "total_frames": total_frames,
            "point_ids": sorted(
                dataframe["point_id"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
            "first_frame": int(dataframe["frame"].min()),
            "last_frame": int(dataframe["frame"].max()),
            "start_time_seconds": float(
                dataframe["time_seconds"].min()
            ),
            "end_time_seconds": float(
                dataframe["time_seconds"].max()
            ),
        },
        "quality_filter": {
            "valid_measurement_count": int(len(valid)),
            "valid_measurement_frames": (
                valid_measurement_frames
            ),
            "retained_timeseries_frames": retained_frames,
            "retained_frame_fraction": (
                retained_frames / total_frames
                if total_frames > 0
                else None
            ),
        },
        "timebase": {
            "median_time_step_seconds": median_time_step,
            "estimated_fps": estimated_fps,
        },
        "point_references": references.to_dict(
            orient="records"
        ),
        "point_statistics": calculate_point_statistics(
            dataframe,
            valid,
            references,
        ),
        "output_columns": list(timeseries.columns),
    }


def write_json(path: Path, content: dict[str, object]) -> None:
    """Schreibt ein Dictionary formatiert als JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            content,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# Konsolenausgabe
# ============================================================

def print_summary(
    *,
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    valid: pd.DataFrame,
    references: pd.DataFrame,
    timeseries: pd.DataFrame,
) -> None:
    """Gibt die wichtigsten Verarbeitungsergebnisse aus."""

    total_frames = int(dataframe["frame"].nunique())
    retained_frames = int(timeseries["frame"].nunique())

    print()
    print("=" * 72)
    print("Lucas-Kanade-Validierungszeitreihe")
    print("=" * 72)
    print(f"Eingabe:                 {args.input_csv}")
    print(f"Ausgabe:                 {args.output_csv}")
    print(f"Zusammenfassung:         {args.summary_json}")
    print(f"Rohdatenzeilen:          {len(dataframe)}")
    print(f"Gültige Punktmessungen:  {len(valid)}")
    print(f"Frames insgesamt:        {total_frames}")
    print(f"Frames übernommen:       {retained_frames}")
    print(
        "Übernommener Anteil:     "
        f"{100.0 * retained_frames / total_frames:.2f} %"
    )
    print(
        "Mindestanzahl Punkte:    "
        f"{args.min_valid_points}"
    )
    print(
        "Referenzwerte pro Punkt: "
        f"{args.reference_samples}"
    )
    print()

    print("Individuelle Punktreferenzen:")

    for row in references.itertuples(index=False):
        print(
            f"  point_id={row.point_id}: "
            f"x_ref={row.reference_x:.6f} px, "
            f"y_ref={row.reference_y:.6f} px, "
            f"n={row.reference_sample_count}"
        )

    print()
    print("Ausgabespalten:")

    for column in timeseries.columns:
        print(f"  - {column}")

    print("=" * 72)


# ============================================================
# Hauptprogramm
# ============================================================

def main() -> None:
    """Führt die vollständige Zeitreihenerzeugung aus."""

    args = parse_args()
    validate_parameters(args)

    dataframe = load_input_csv(args.input_csv)

    valid = filter_valid_measurements(
        dataframe,
        max_fb_error=args.max_fb_error,
        max_jump_px=args.max_jump_px,
        selected_point_ids=args.point_ids,
    )

    references = calculate_point_references(
        valid,
        reference_samples=args.reference_samples,
    )

    pointwise = add_pointwise_displacements(
        valid,
        references,
    )

    timeseries = aggregate_frames(
        pointwise,
        min_valid_points=args.min_valid_points,
    )

    diagnostics = create_point_diagnostic_columns(
        pointwise
    )

    timeseries = merge_diagnostics(
        timeseries,
        diagnostics,
    )

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeseries.to_csv(
        args.output_csv,
        index=False,
    )

    summary = build_summary(
        args=args,
        dataframe=dataframe,
        valid=valid,
        references=references,
        timeseries=timeseries,
    )

    write_json(
        args.summary_json,
        summary,
    )

    print_summary(
        args=args,
        dataframe=dataframe,
        valid=valid,
        references=references,
        timeseries=timeseries,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nVerarbeitung durch Benutzer abgebrochen.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nFEHLER: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc