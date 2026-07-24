"""Robuster IMU-Loader für archivierte Messdaten.

Unterstützt numerische Zeitspalten sowie die Kombination ``day`` + ``time``.
Die bestehende Datei ``imu_loader.py`` kann nach Prüfung durch diese Version
ersetzt werden.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

IMU_AXES = ("linx", "liny", "linz", "rotx", "roty", "rotz")

class IMULoaderError(Exception):
    """Raised when IMU data cannot be loaded."""


def _relative_time_seconds(df: pd.DataFrame, requested: Optional[str] = None) -> tuple[np.ndarray, str]:
    candidates = [requested] if requested else []
    candidates += ["time_seconds", "timestamp", "writetime", "time", "t"]

    if requested in (None, "time", "day+time") and {"day", "time"}.issubset(df.columns):
        combined = df["day"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        timestamps = pd.to_datetime(combined, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        if timestamps.notna().sum() < 2:
            timestamps = pd.to_datetime(combined, errors="coerce")
        if timestamps.notna().sum() >= 2:
            first = timestamps.dropna().iloc[0]
            return (timestamps - first).dt.total_seconds().to_numpy(float), "day+time"

    for column in dict.fromkeys(column for column in candidates if column):
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() >= 2:
            values = numeric.to_numpy(float)
            first = values[np.flatnonzero(np.isfinite(values))[0]]
            return values - first, column
        timestamps = pd.to_datetime(df[column], errors="coerce")
        if timestamps.notna().sum() >= 2:
            first = timestamps.dropna().iloc[0]
            return (timestamps - first).dt.total_seconds().to_numpy(float), column

    raise IMULoaderError("No usable IMU time column found.")


def load_imu_signal(
    csv_file: str | Path,
    axis: str = "liny",
    time_column: Optional[str] = None,
    value_column: Optional[str] = None,
    signal_column: Optional[str] = None,
    sample_rate_hz: Optional[float] = None,
    **_: object,
) -> dict:
    path = Path(csv_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    selected_axis = value_column or signal_column or axis
    if selected_axis not in df.columns:
        raise IMULoaderError(f"Axis '{selected_axis}' not found. Available: {list(df.columns)}")

    try:
        time, used_time_column = _relative_time_seconds(df, time_column)
    except IMULoaderError:
        if sample_rate_hz is None or sample_rate_hz <= 0:
            raise
        time = np.arange(len(df), dtype=float) / float(sample_rate_hz)
        used_time_column = "generated_from_sample_rate"

    signal = pd.to_numeric(df[selected_axis], errors="coerce").to_numpy(float)
    mask = np.isfinite(time) & np.isfinite(signal)
    time, signal = time[mask], signal[mask]
    order = np.argsort(time, kind="stable")
    time, signal = time[order], signal[order]
    unique, indices = np.unique(time, return_index=True)
    time, signal = unique, signal[indices]
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise IMULoaderError("Not enough valid, strictly increasing samples.")
    rate = float(1.0 / np.median(np.diff(time)))
    return {
        "time": time,
        "signal": signal,
        "sampling_rate": rate,
        "duration": float(time[-1] - time[0]),
        "num_samples": int(len(signal)),
        "signal_name": selected_axis,
        "time_column": used_time_column,
        "source": str(path),
    }
