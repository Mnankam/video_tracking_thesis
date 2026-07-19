"""
synchronization.py

Automatic synchronization using cross correlation.

Author: Serge Kouomnankam
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate


def estimate_time_shift(
    reference_signal,
    target_signal,
    sampling_rate,
):
    """
    Estimate time shift using normalized cross correlation.

    Returns
    -------
    dict
    """

    reference_signal = np.asarray(reference_signal, dtype=float)
    target_signal = np.asarray(target_signal, dtype=float)

    reference_signal = reference_signal - np.mean(reference_signal)
    target_signal = target_signal - np.mean(target_signal)

    correlation = correlate(
        target_signal,
        reference_signal,
        mode="full",
    )

    lags = np.arange(
        -len(reference_signal) + 1,
        len(target_signal),
    )

    best_index = np.argmax(correlation)

    lag_samples = int(lags[best_index])

    lag_seconds = lag_samples / sampling_rate

    return {
        "lag_samples": lag_samples,
        "lag_seconds": lag_seconds,
        "correlation": correlation,
        "lags": lags,
    }


def synchronize_signals(
    video_time,
    video_signal,
    imu_time,
    imu_signal,
    sampling_rate,
):
    """
    Synchronize IMU to video signal.

    Returns
    -------
    dict
    """

    result = estimate_time_shift(
        video_signal,
        imu_signal,
        sampling_rate,
    )

    lag = result["lag_samples"]

    if lag > 0:

        video_sync = video_signal

        imu_sync = imu_signal[lag:]

    elif lag < 0:

        video_sync = video_signal[-lag:]

        imu_sync = imu_signal

    else:

        video_sync = video_signal

        imu_sync = imu_signal

    n = min(len(video_sync), len(imu_sync))

    video_sync = video_sync[:n]
    imu_sync = imu_sync[:n]

    time_sync = np.arange(n) / sampling_rate

    return {
        "time": time_sync,
        "video_signal": video_sync,
        "imu_signal": imu_sync,
        "lag_samples": result["lag_samples"],
        "lag_seconds": result["lag_seconds"],
        "cross_correlation": result["correlation"],
        "lags": result["lags"],
    }