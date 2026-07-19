"""
metrics.py

Quantitative metrics for Video ↔ IMU validation.

Author: Serge Kouomnankam
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window


class MetricsError(ValueError):
    """Raised when validation metrics cannot be calculated."""


@dataclass(frozen=True)
class FrequencyPeak:
    rank: int
    frequency_hz: float
    amplitude: float


@dataclass(frozen=True)
class ValidationMetrics:
    num_samples: int
    sampling_rate_hz: float
    duration_s: float
    pearson_correlation: float
    rmse: float
    normalized_rmse: float
    mae: float
    max_absolute_error: float
    r_squared: float
    video_dominant_frequency_hz: float
    imu_dominant_frequency_hz: float
    dominant_frequency_error_hz: float
    dominant_frequency_error_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_finite_1d_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise MetricsError(f"'{name}' must be one-dimensional.")
    if array.size < 2:
        raise MetricsError(f"'{name}' must contain at least two samples.")
    if not np.all(np.isfinite(array)):
        raise MetricsError(f"'{name}' contains NaN or infinite values.")
    return array


def _validate_signal_pair(
    video_signal: Any,
    imu_signal: Any,
) -> tuple[np.ndarray, np.ndarray]:
    video = _as_finite_1d_array(video_signal, "video_signal")
    imu = _as_finite_1d_array(imu_signal, "imu_signal")
    if video.size != imu.size:
        raise MetricsError(
            "The synchronized signals must have equal lengths. "
            f"Video: {video.size}, IMU: {imu.size}."
        )
    return video, imu


def calculate_pearson_correlation(
    video_signal: Any,
    imu_signal: Any,
) -> float:
    video, imu = _validate_signal_pair(video_signal, imu_signal)
    if np.isclose(np.std(video), 0.0) or np.isclose(np.std(imu), 0.0):
        return float("nan")
    return float(np.corrcoef(video, imu)[0, 1])


def calculate_rmse(video_signal: Any, imu_signal: Any) -> float:
    video, imu = _validate_signal_pair(video_signal, imu_signal)
    return float(np.sqrt(np.mean((video - imu) ** 2)))


def calculate_normalized_rmse(
    video_signal: Any,
    imu_signal: Any,
) -> float:
    video, imu = _validate_signal_pair(video_signal, imu_signal)
    imu_range = float(np.max(imu) - np.min(imu))
    if np.isclose(imu_range, 0.0):
        return float("nan")
    return float(calculate_rmse(video, imu) / imu_range)


def calculate_mae(video_signal: Any, imu_signal: Any) -> float:
    video, imu = _validate_signal_pair(video_signal, imu_signal)
    return float(np.mean(np.abs(video - imu)))


def calculate_max_absolute_error(
    video_signal: Any,
    imu_signal: Any,
) -> float:
    video, imu = _validate_signal_pair(video_signal, imu_signal)
    return float(np.max(np.abs(video - imu)))


def calculate_r_squared(video_signal: Any, imu_signal: Any) -> float:
    video, imu = _validate_signal_pair(video_signal, imu_signal)
    residual_sum = float(np.sum((imu - video) ** 2))
    total_sum = float(np.sum((imu - np.mean(imu)) ** 2))
    if np.isclose(total_sum, 0.0):
        return float("nan")
    return float(1.0 - residual_sum / total_sum)


def compute_fft_spectrum(
    signal: Any,
    sampling_rate: float,
    *,
    window: str = "hann",
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_finite_1d_array(signal, "signal")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise MetricsError("sampling_rate must be a positive finite value.")

    if remove_mean:
        values = values - np.mean(values)

    weights = get_window(window, values.size, fftbins=True)
    weighted_signal = values * weights
    fft_values = np.fft.rfft(weighted_signal)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate)

    coherent_gain = float(np.sum(weights) / values.size)
    if np.isclose(coherent_gain, 0.0):
        raise MetricsError("The selected FFT window has zero coherent gain.")

    amplitudes = np.abs(fft_values) / (values.size * coherent_gain)
    if amplitudes.size > 1:
        if values.size % 2 == 0:
            amplitudes[1:-1] *= 2.0
        else:
            amplitudes[1:] *= 2.0

    return frequencies, amplitudes


def find_dominant_frequency(
    signal: Any,
    sampling_rate: float,
    *,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    window: str = "hann",
) -> tuple[float, float]:
    frequencies, amplitudes = compute_fft_spectrum(
        signal,
        sampling_rate,
        window=window,
        remove_mean=True,
    )

    nyquist = sampling_rate / 2.0
    upper_limit = (
        nyquist
        if max_frequency_hz is None
        else min(float(max_frequency_hz), nyquist)
    )
    lower_limit = max(float(min_frequency_hz), 0.0)

    mask = (frequencies >= lower_limit) & (frequencies <= upper_limit)
    if lower_limit <= 0.0:
        mask &= frequencies > 0.0

    if not np.any(mask):
        raise MetricsError(
            "No FFT bins are available inside the requested frequency range."
        )

    selected_indices = np.flatnonzero(mask)
    best_local_index = int(np.argmax(amplitudes[mask]))
    best_index = int(selected_indices[best_local_index])

    return float(frequencies[best_index]), float(amplitudes[best_index])


def find_spectral_peaks(
    signal: Any,
    sampling_rate: float,
    *,
    number_of_peaks: int = 5,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    minimum_prominence: float | None = None,
    window: str = "hann",
) -> list[FrequencyPeak]:
    if number_of_peaks < 1:
        raise MetricsError("number_of_peaks must be at least 1.")

    frequencies, amplitudes = compute_fft_spectrum(
        signal,
        sampling_rate,
        window=window,
        remove_mean=True,
    )

    nyquist = sampling_rate / 2.0
    upper_limit = (
        nyquist
        if max_frequency_hz is None
        else min(float(max_frequency_hz), nyquist)
    )

    mask = (
        (frequencies >= max(float(min_frequency_hz), 0.0))
        & (frequencies <= upper_limit)
        & (frequencies > 0.0)
    )

    selected_frequencies = frequencies[mask]
    selected_amplitudes = amplitudes[mask]

    if selected_frequencies.size == 0:
        return []

    peak_indices, _ = find_peaks(
        selected_amplitudes,
        prominence=minimum_prominence,
    )

    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(selected_amplitudes))])

    sorted_indices = peak_indices[
        np.argsort(selected_amplitudes[peak_indices])[::-1]
    ][:number_of_peaks]

    return [
        FrequencyPeak(
            rank=rank,
            frequency_hz=float(selected_frequencies[index]),
            amplitude=float(selected_amplitudes[index]),
        )
        for rank, index in enumerate(sorted_indices, start=1)
    ]


def calculate_frequency_error(
    video_frequency_hz: float,
    imu_frequency_hz: float,
) -> tuple[float, float]:
    absolute_error = abs(float(video_frequency_hz) - float(imu_frequency_hz))
    if np.isclose(imu_frequency_hz, 0.0):
        relative_error = float("nan")
    else:
        relative_error = absolute_error / abs(float(imu_frequency_hz)) * 100.0
    return float(absolute_error), float(relative_error)


def compute_validation_metrics(
    time: Any,
    video_signal: Any,
    imu_signal: Any,
    sampling_rate: float,
    *,
    min_frequency_hz: float = 0.5,
    max_frequency_hz: float | None = None,
    fft_window: str = "hann",
) -> ValidationMetrics:
    time_array = _as_finite_1d_array(time, "time")
    video, imu = _validate_signal_pair(video_signal, imu_signal)

    if time_array.size != video.size:
        raise MetricsError(
            "The time vector and synchronized signals must have equal lengths."
        )

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise MetricsError("sampling_rate must be a positive finite value.")

    video_frequency, _ = find_dominant_frequency(
        video,
        sampling_rate,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        window=fft_window,
    )
    imu_frequency, _ = find_dominant_frequency(
        imu,
        sampling_rate,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        window=fft_window,
    )

    frequency_error_hz, frequency_error_percent = calculate_frequency_error(
        video_frequency,
        imu_frequency,
    )

    return ValidationMetrics(
        num_samples=int(video.size),
        sampling_rate_hz=float(sampling_rate),
        duration_s=float(time_array[-1] - time_array[0]),
        pearson_correlation=calculate_pearson_correlation(video, imu),
        rmse=calculate_rmse(video, imu),
        normalized_rmse=calculate_normalized_rmse(video, imu),
        mae=calculate_mae(video, imu),
        max_absolute_error=calculate_max_absolute_error(video, imu),
        r_squared=calculate_r_squared(video, imu),
        video_dominant_frequency_hz=video_frequency,
        imu_dominant_frequency_hz=imu_frequency,
        dominant_frequency_error_hz=frequency_error_hz,
        dominant_frequency_error_percent=frequency_error_percent,
    )


def metrics_to_dataframe(
    metrics: ValidationMetrics | dict[str, Any],
) -> pd.DataFrame:
    values = metrics.to_dict() if isinstance(metrics, ValidationMetrics) else metrics
    return pd.DataFrame(
        {
            "metric": list(values.keys()),
            "value": list(values.values()),
        }
    )


def save_metrics_csv(
    metrics: ValidationMetrics | dict[str, Any],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(metrics).to_csv(output_path, index=False)
    return output_path