"""
preprocessing.py

Signal preprocessing for Video ↔ IMU validation.

Author: Serge Kouomnankam
"""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend
from scipy.interpolate import interp1d


def remove_offset(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset."""
    return signal - np.mean(signal)


def remove_linear_trend(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend."""
    return detrend(signal, type="linear")


def normalize(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize signal.

    Parameters
    ----------
    method
        zscore
        minmax
        none
    """

    signal = np.asarray(signal, dtype=float)

    if method.lower() == "none":
        return signal.copy()

    if method.lower() == "zscore":

        std = np.std(signal)

        if std == 0:
            return signal.copy()

        return (signal - np.mean(signal)) / std

    if method.lower() == "minmax":

        minimum = np.min(signal)
        maximum = np.max(signal)

        if maximum == minimum:
            return signal.copy()

        return (signal - minimum) / (maximum - minimum)

    raise ValueError(f"Unknown normalization method: {method}")


def resample_signal(
    time: np.ndarray,
    signal: np.ndarray,
    target_fs: float,
):
    """
    Resample signal onto a uniform time grid.

    Returns
    -------
    new_time
    new_signal
    """

    dt = 1.0 / target_fs

    new_time = np.arange(
        time[0],
        time[-1],
        dt,
    )

    interpolator = interp1d(
        time,
        signal,
        kind="linear",
        fill_value="extrapolate",
    )

    new_signal = interpolator(new_time)

    return new_time, new_signal


def preprocess_signal(
    time,
    signal,
    target_fs=None,
    detrend_signal=True,
    normalization="zscore",
):
    """
    Complete preprocessing pipeline.

    Returns
    -------
    dict
    """

    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    mask = np.isfinite(time) & np.isfinite(signal)

    time = time[mask]
    signal = signal[mask]

    signal = remove_offset(signal)

    if detrend_signal:
        signal = remove_linear_trend(signal)

    signal = normalize(signal, normalization)

    if target_fs is not None:

        time, signal = resample_signal(
            time,
            signal,
            target_fs,
        )

        sampling_rate = target_fs

    else:

        sampling_rate = 1.0 / np.median(np.diff(time))

    return {
        "time": time,
        "signal": signal,
        "sampling_rate": sampling_rate,
    }