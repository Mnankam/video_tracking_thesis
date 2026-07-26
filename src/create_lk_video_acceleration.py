#!/usr/bin/env python3
"""
create_lk_video_acceleration.py
===============================

Erzeugt aus den Lucas-Kanade-Punkttrajektorien eine robuste videobasierte
Position, Geschwindigkeit und Beschleunigung.

Da keine metrische Bildkalibrierung verfügbar ist, werden die Größen in

    Position:       px
    Geschwindigkeit: px/s
    Beschleunigung:  px/s²

ausgegeben.

Die Beschleunigung kann nach Z-Normalisierung mit den linearen IMU-Achsen
linx, liny und linz verglichen werden. Ein absoluter Amplitudenvergleich
in m/s² ist ohne Meter-pro-Pixel-Kalibrierung nicht zulässig.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


INPUT_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/Lucas_Kanade_CPU_1/"
    "GX010262_lucas_kanade.csv"
)

OUTPUT_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/Lucas_Kanade_CPU_1/"
    "GX010262_lk_video_acceleration_px.csv"
)

# Punkte 0, 1 und 2 wurden als innere Rohr-/Reflexionsstruktur definiert.
POINT_IDS = [0, 1, 2]

MAX_FB_ERROR = 1.0
MAX_JUMP_PX = 5.0
REFERENCE_SAMPLES = 20
MIN_VALID_POINTS = 2

VIDEO_FPS = 200.0

# Savitzky-Golay-Parameter.
# 61 Samples bei 200 FPS entsprechen 0,305 Sekunden.
SAVGOL_WINDOW = 61
SAVGOL_POLYORDER = 3

# Bewegungsachse: "x" oder "y".
MOTION_AXIS = "x"


def _validate_input(raw: pd.DataFrame) -> None:
    required = {
        "frame",
        "time_seconds",
        "point_id",
        "x",
        "y",
        "jump_px",
        "fb_error",
        "tracking_status",
    }

    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            "In der Lucas-Kanade-Datei fehlen Spalten: "
            f"{sorted(missing)}"
        )


def _prepare_savgol_window(
    number_of_samples: int,
    requested_window: int,
    polynomial_order: int,
) -> int:
    """Erzeugt eine gültige ungerade Savitzky-Golay-Fensterlänge."""

    window = int(requested_window)

    if window % 2 == 0:
        window += 1

    if window >= number_of_samples:
        window = number_of_samples - 1

    if window % 2 == 0:
        window -= 1

    if window <= polynomial_order:
        raise ValueError(
            "Zu wenige Daten für den Savitzky-Golay-Filter: "
            f"Fenster={window}, Polynomordnung={polynomial_order}."
        )

    return window


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardisiert ein Signal auf Mittelwert 0 und Standardabweichung 1."""

    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    standard_deviation = float(np.nanstd(values))

    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        return np.full_like(values, np.nan)

    return (values - mean) / standard_deviation


def main() -> None:
    raw = pd.read_csv(INPUT_CSV)
    _validate_input(raw)

    print(f"Eingabedatei: {INPUT_CSV}")
    print(f"Rohzeilen: {len(raw)}")

    # Nur die gewünschte physische Punktgruppe verwenden.
    selected = raw.loc[
        raw["point_id"].isin(POINT_IDS)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"Keine Daten für die Punkt-IDs {POINT_IDS} vorhanden."
        )

    # Nur qualitativ gültige Punktmessungen behalten.
    selected = selected.loc[
        (pd.to_numeric(selected["tracking_status"], errors="coerce") == 1)
        & (
            pd.to_numeric(selected["fb_error"], errors="coerce")
            <= MAX_FB_ERROR
        )
        & (
            pd.to_numeric(selected["jump_px"], errors="coerce")
            <= MAX_JUMP_PX
        )
    ].copy()

    selected = selected.dropna(
        subset=[
            "frame",
            "time_seconds",
            "point_id",
            "x",
            "y",
            "fb_error",
            "jump_px",
        ]
    )

    if selected.empty:
        raise ValueError(
            "Nach Qualitätsfilterung bleiben keine Lucas-Kanade-Daten übrig."
        )

    print(f"Gültige Punktmessungen: {len(selected)}")

    # Referenzposition pro Punkt aus den ersten gültigen Samples.
    references = (
        selected
        .sort_values(["point_id", "time_seconds"], kind="stable")
        .groupby("point_id", group_keys=False)
        .head(REFERENCE_SAMPLES)
        .groupby("point_id")
        .agg(
            x_reference=("x", "median"),
            y_reference=("y", "median"),
            reference_sample_count=("x", "count"),
        )
    )

    missing_reference_points = [
        point_id
        for point_id in POINT_IDS
        if point_id not in references.index
    ]

    if missing_reference_points:
        print(
            "Warnung: Keine gültige Referenz für Punkte: "
            f"{missing_reference_points}"
        )

    selected = selected.join(references, on="point_id")

    selected = selected.dropna(
        subset=["x_reference", "y_reference"]
    )

    selected["point_displacement_x_px"] = (
        selected["x"] - selected["x_reference"]
    )

    selected["point_displacement_y_px"] = (
        selected["y"] - selected["y_reference"]
    )

    # Pro Frame robuste Aggregation über die gewählte Punktgruppe.
    grouped = (
        selected
        .groupby(["frame", "time_seconds"], as_index=False)
        .agg(
            displacement_x_px=(
                "point_displacement_x_px",
                "median",
            ),
            displacement_y_px=(
                "point_displacement_y_px",
                "median",
            ),
            valid_point_count=("point_id", "nunique"),
            median_fb_error=("fb_error", "median"),
            max_fb_error=("fb_error", "max"),
            median_jump_px=("jump_px", "median"),
            max_jump_px=("jump_px", "max"),
            displacement_x_std=(
                "point_displacement_x_px",
                "std",
            ),
            displacement_y_std=(
                "point_displacement_y_px",
                "std",
            ),
            displacement_x_min=(
                "point_displacement_x_px",
                "min",
            ),
            displacement_x_max=(
                "point_displacement_x_px",
                "max",
            ),
            displacement_y_min=(
                "point_displacement_y_px",
                "min",
            ),
            displacement_y_max=(
                "point_displacement_y_px",
                "max",
            ),
        )
        .sort_values("time_seconds", kind="stable")
        .reset_index(drop=True)
    )

    grouped["displacement_x_range_px"] = (
        grouped["displacement_x_max"]
        - grouped["displacement_x_min"]
    )

    grouped["displacement_y_range_px"] = (
        grouped["displacement_y_max"]
        - grouped["displacement_y_min"]
    )

    grouped = grouped.loc[
        grouped["valid_point_count"] >= MIN_VALID_POINTS
    ].copy()

    if len(grouped) < SAVGOL_WINDOW:
        raise ValueError(
            "Nach Aggregation bleiben zu wenige Frames für die Ableitung: "
            f"{len(grouped)}."
        )

    print(f"Gültige aggregierte Frames: {len(grouped)}")
    print(
        "Mittlere Anzahl gültiger Punkte pro Frame: "
        f"{grouped['valid_point_count'].mean():.3f}"
    )

    # Gleichmäßige Videozeitachse.
    dt = 1.0 / VIDEO_FPS

    uniform_time = np.arange(
        float(grouped["time_seconds"].iloc[0]),
        float(grouped["time_seconds"].iloc[-1])
        + 0.25 * dt,
        dt,
    )

    displacement_x_px = np.interp(
        uniform_time,
        grouped["time_seconds"].to_numpy(float),
        grouped["displacement_x_px"].to_numpy(float),
    )

    displacement_y_px = np.interp(
        uniform_time,
        grouped["time_seconds"].to_numpy(float),
        grouped["displacement_y_px"].to_numpy(float),
    )

    valid_point_count = np.interp(
        uniform_time,
        grouped["time_seconds"].to_numpy(float),
        grouped["valid_point_count"].to_numpy(float),
    )

    median_fb_error = np.interp(
        uniform_time,
        grouped["time_seconds"].to_numpy(float),
        grouped["median_fb_error"].to_numpy(float),
    )

    displacement_x_range_px = np.interp(
        uniform_time,
        grouped["time_seconds"].to_numpy(float),
        grouped["displacement_x_range_px"]
        .fillna(0.0)
        .to_numpy(float),
    )

    displacement_y_range_px = np.interp(
        uniform_time,
        grouped["time_seconds"].to_numpy(float),
        grouped["displacement_y_range_px"]
        .fillna(0.0)
        .to_numpy(float),
    )

    if MOTION_AXIS == "x":
        position_px = displacement_x_px
    elif MOTION_AXIS == "y":
        position_px = displacement_y_px
    else:
        raise ValueError(
            f"MOTION_AXIS muss 'x' oder 'y' sein, nicht {MOTION_AXIS!r}."
        )

    window = _prepare_savgol_window(
        number_of_samples=len(position_px),
        requested_window=SAVGOL_WINDOW,
        polynomial_order=SAVGOL_POLYORDER,
    )

    # Geglättete Position.
    position_filtered_px = savgol_filter(
        position_px,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        deriv=0,
        delta=dt,
        mode="interp",
    )

    # Erste Ableitung.
    velocity_px_s = savgol_filter(
        position_px,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        deriv=1,
        delta=dt,
        mode="interp",
    )

    # Zweite Ableitung.
    acceleration_px_s2 = savgol_filter(
        position_px,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        deriv=2,
        delta=dt,
        mode="interp",
    )

    # Zusätzliche standardisierte Beschleunigung für direkte Formvergleiche.
    acceleration_zscore = _zscore(acceleration_px_s2)

    output = pd.DataFrame(
        {
            "time_seconds": uniform_time,
            "lk_position_x_px": displacement_x_px,
            "lk_position_y_px": displacement_y_px,
            "lk_position_selected_px": position_px,
            "lk_position_filtered_px": position_filtered_px,
            "lk_velocity_px_s": velocity_px_s,
            "lk_acceleration_px_s2": acceleration_px_s2,
            "lk_acceleration_zscore": acceleration_zscore,
            "valid_point_count_interpolated": valid_point_count,
            "median_fb_error_interpolated": median_fb_error,
            "displacement_x_range_px": displacement_x_range_px,
            "displacement_y_range_px": displacement_y_range_px,
        }
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(f"Ausgabe: {OUTPUT_CSV}")
    print(f"Zeilen: {len(output)}")
    print(
        "Zeitbereich: "
        f"{uniform_time[0]:.3f} bis "
        f"{uniform_time[-1]:.3f} s"
    )
    print(f"Video-FPS: {VIDEO_FPS:.3f}")
    print(f"Savitzky-Golay-Fenster: {window} Samples")
    print(
        "Filterdauer: "
        f"{window / VIDEO_FPS:.3f} s"
    )
    print(
        "Beschleunigungsbereich: "
        f"{np.nanmin(acceleration_px_s2):.6g} bis "
        f"{np.nanmax(acceleration_px_s2):.6g} px/s²"
    )

    print()
    print(
        "Hinweis: Ohne geometrische Kalibrierung ist kein absoluter "
        "Amplitudenvergleich in m/s² möglich."
    )
    print(
        "Korrelation, Kreuzkorrelation, Frequenz, Phase und normierte "
        "Signalform können trotzdem ausgewertet werden."
    )


if __name__ == "__main__":
    main()