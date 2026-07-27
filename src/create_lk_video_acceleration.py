#!/usr/bin/env python3

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
    "GX010262_lk_video_acceleration_m/s^2.csv"
)

# Zunächst nur die als Rohr-/Reflexionsstruktur definierten Punkte.
POINT_IDS = [0, 1, 2]

MAX_FB_ERROR = 1.0
MAX_JUMP_PX = 5.0
REFERENCE_SAMPLES = 20
MIN_VALID_POINTS = 2

# MUSS anhand einer bekannten Länge im Bild bestimmt werden.
METER_PER_PIXEL = 0.0004  

SAVGOL_WINDOW = 61
SAVGOL_POLYORDER = 3


def main() -> None:
    raw = pd.read_csv(INPUT_CSV)

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
        raise ValueError(f"Fehlende Spalten: {sorted(missing)}")

    selected = raw.loc[raw["point_id"].isin(POINT_IDS)].copy()

    selected = selected.loc[
        (selected["tracking_status"] == 1)
        & (selected["fb_error"] <= MAX_FB_ERROR)
        & (selected["jump_px"] <= MAX_JUMP_PX)
    ].copy()

    if selected.empty:
        raise ValueError("Nach Qualitätsfilterung bleiben keine Daten übrig.")

    references = (
        selected.sort_values(["point_id", "time_seconds"])
        .groupby("point_id", group_keys=False)
        .head(REFERENCE_SAMPLES)
        .groupby("point_id")
        .agg(
            x_reference=("x", "median"),
            y_reference=("y", "median"),
        )
    )

    selected = selected.join(references, on="point_id")

    selected["point_displacement_x_px"] = (
        selected["x"] - selected["x_reference"]
    )
    selected["point_displacement_y_px"] = (
        selected["y"] - selected["y_reference"]
    )

    grouped = (
        selected.groupby(["frame", "time_seconds"], as_index=False)
        .agg(
            displacement_x_px=("point_displacement_x_px", "median"),
            displacement_y_px=("point_displacement_y_px", "median"),
            valid_point_count=("point_id", "nunique"),
            median_fb_error=("fb_error", "median"),
            displacement_x_std=("point_displacement_x_px", "std"),
            displacement_y_std=("point_displacement_y_px", "std"),
        )
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )

    grouped = grouped.loc[
        grouped["valid_point_count"] >= MIN_VALID_POINTS
    ].copy()

    # Einheitliche 200-Hz-Zeitachse herstellen.
    dt = 1.0 / 200.0
    uniform_time = np.arange(
        grouped["time_seconds"].iloc[0],
        grouped["time_seconds"].iloc[-1] + 0.25 * dt,
        dt,
    )

    displacement_x_px = np.interp(
        uniform_time,
        grouped["time_seconds"],
        grouped["displacement_x_px"],
    )
    displacement_y_px = np.interp(
        uniform_time,
        grouped["time_seconds"],
        grouped["displacement_y_px"],
    )

    # Hier zunächst X als angenommene Bewegungsrichtung.
    position_px = displacement_x_px
    position_m = position_px * METER_PER_PIXEL

    window = SAVGOL_WINDOW
    if window % 2 == 0:
        window += 1
    if window >= len(position_m):
        window = len(position_m) - 1
        if window % 2 == 0:
            window -= 1

    if window <= SAVGOL_POLYORDER:
        raise ValueError("Zu wenige Daten für den Savitzky-Golay-Filter.")

    position_filtered_m = savgol_filter(
        position_m,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        deriv=0,
        delta=dt,
    )

    velocity_m_s = savgol_filter(
        position_m,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        deriv=1,
        delta=dt,
    )

    acceleration_m_s2 = savgol_filter(
        position_m,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        deriv=2,
        delta=dt,
    )

    output = pd.DataFrame(
        {
            "time_seconds": uniform_time,
            "lk_position_x_px": position_px,
            "lk_position_x_m": position_m,
            "lk_position_x_filtered_m": position_filtered_m,
            "lk_velocity_x_m_s": velocity_m_s,
            "lk_acceleration_x_m_s2": acceleration_m_s2,
        }
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"Ausgabe: {OUTPUT_CSV}")
    print(f"Zeilen: {len(output)}")
    print(f"Zeitbereich: {uniform_time[0]:.3f} bis {uniform_time[-1]:.3f} s")
    print(
        "Beschleunigungsbereich:",
        f"{acceleration_m_s2.min():.6g}",
        "bis",
        f"{acceleration_m_s2.max():.6g} m/s²",
    )


if __name__ == "__main__":
    main()