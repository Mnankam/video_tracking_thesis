#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/Lucas_Kanade_CPU8/"
    "GX010262_lucas_kanade.csv"
)

OUTPUT_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/Lucas_Kanade_CPU8/"
    "GX010262_lucas_kanade_timeseries.csv"
)

MAX_FB_ERROR = 1.0
MAX_JUMP_PX = 20.0


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
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

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Fehlende Spalten: {sorted(missing)}\n"
            f"Vorhandene Spalten: {list(df.columns)}"
        )

    numeric_columns = [
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

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "frame",
            "time_seconds",
            "point_id",
            "x",
            "y",
            "tracking_status",
        ]
    ).copy()

    # Nur erfolgreiche und ausreichend zuverlässige Trackings.
    valid = df[
        (df["tracking_status"] == 1)
        & np.isfinite(df["x"])
        & np.isfinite(df["y"])
        & np.isfinite(df["fb_error"])
        & np.isfinite(df["jump_px"])
        & (df["fb_error"] <= MAX_FB_ERROR)
        & (df["jump_px"] <= MAX_JUMP_PX)
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "Nach der Qualitätsfilterung sind keine gültigen "
            "Lucas-Kanade-Punkte übrig."
        )

    # Genau ein robuster Messwert pro Frame.
    timeseries = (
        valid.groupby(["frame", "time_seconds"], as_index=False)
        .agg(
            lk_center_x=("x", "median"),
            lk_center_y=("y", "median"),
            lk_median_dx=("dx", "median"),
            lk_median_dy=("dy", "median"),
            lk_median_jump_px=("jump_px", "median"),
            lk_median_fb_error=("fb_error", "median"),
            valid_point_count=("point_id", "nunique"),
        )
        .sort_values(["time_seconds", "frame"])
        .reset_index(drop=True)
    )

    if timeseries.empty:
        raise RuntimeError("Es konnte keine Zeitreihe erzeugt werden.")

    # Bewegung relativ zum ersten robusten Positionswert.
    x_reference = float(timeseries["lk_center_x"].iloc[0])
    y_reference = float(timeseries["lk_center_y"].iloc[0])

    timeseries["lk_displacement_x"] = (
        timeseries["lk_center_x"] - x_reference
    )
    timeseries["lk_displacement_y"] = (
        timeseries["lk_center_y"] - y_reference
    )

    # Optional: resultierende zweidimensionale Verschiebung.
    timeseries["lk_displacement_magnitude"] = np.sqrt(
        timeseries["lk_displacement_x"] ** 2
        + timeseries["lk_displacement_y"] ** 2
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(OUTPUT_CSV, index=False)

    total_frames = int(df["frame"].nunique())
    retained_frames = int(timeseries["frame"].nunique())

    print(f"Eingabe:            {INPUT_CSV}")
    print(f"Ausgabe:            {OUTPUT_CSV}")
    print(f"Frames insgesamt:   {total_frames}")
    print(f"Frames übernommen:  {retained_frames}")
    print(f"Ausgabezeilen:      {len(timeseries)}")
    print(f"X-Referenz:         {x_reference:.6f} px")
    print(f"Y-Referenz:         {y_reference:.6f} px")
    print("\nAusgabespalten:")
    for column in timeseries.columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()