"""
plotting.py

Plot generation for Video ↔ IMU validation.

Author: Serge Kouomnankam
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from .metrics import compute_fft_spectrum

import matplotlib

# Required for execution on headless cluster nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import MetricsError, compute_fft_spectrum


class PlottingError(ValueError):
    """Raised when a validation plot cannot be generated."""


def _as_finite_1d_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise PlottingError(f"'{name}' must be one-dimensional.")

    if array.size < 2:
        raise PlottingError(f"'{name}' must contain at least two samples.")

    if not np.all(np.isfinite(array)):
        raise PlottingError(f"'{name}' contains NaN or infinite values.")

    return array


def _validate_time_domain_data(
    time: Any,
    video_signal: Any,
    imu_signal: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_array = _as_finite_1d_array(time, "time")
    video = _as_finite_1d_array(video_signal, "video_signal")
    imu = _as_finite_1d_array(imu_signal, "imu_signal")

    if not (time_array.size == video.size == imu.size):
        raise PlottingError(
            "Time, video signal and IMU signal must have equal lengths."
        )

    return time_array, video, imu


def _prepare_output_file(output_file: str | Path) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_and_close(
    figure: plt.Figure,
    output_file: str | Path,
    dpi: int,
) -> Path:
    output_path = _prepare_output_file(output_file)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_time_domain_comparison(
    time: Any,
    video_signal: Any,
    imu_signal: Any,
    output_file: str | Path,
    *,
    video_label: str = "Video",
    imu_label: str = "IMU",
    title: str = "Synchronized video and IMU signals",
    y_label: str = "Normalized amplitude",
    dpi: int = 300,
) -> Path:
    time_array, video, imu = _validate_time_domain_data(
        time,
        video_signal,
        imu_signal,
    )

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(time_array, video, linewidth=1.0, label=video_label)
    axis.plot(time_array, imu, linewidth=1.0, alpha=0.8, label=imu_label)

    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    return _save_and_close(figure, output_file, dpi)


def plot_residual_error(
    time: Any,
    video_signal: Any,
    imu_signal: Any,
    output_file: str | Path,
    *,
    title: str = "Residual error between video and IMU",
    dpi: int = 300,
) -> Path:
    time_array, video, imu = _validate_time_domain_data(
        time,
        video_signal,
        imu_signal,
    )
    residual = video - imu

    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(time_array, residual, linewidth=0.9)
    axis.axhline(0.0, linewidth=1.0, linestyle="--")

    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Residual amplitude")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    return _save_and_close(figure, output_file, dpi)


def plot_fft_comparison(
    video_signal: Any,
    imu_signal: Any,
    sampling_rate: float,
    output_file: str | Path,
    *,
    video_label: str = "Video",
    imu_label: str = "IMU",
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    fft_window: str = "hann",
    logarithmic_y: bool = False,
    title: str = "Frequency-domain comparison",
    dpi: int = 300,
) -> Path:
    video = _as_finite_1d_array(video_signal, "video_signal")
    imu = _as_finite_1d_array(imu_signal, "imu_signal")

    if video.size != imu.size:
        raise PlottingError(
            "The synchronized video and IMU signals must have equal lengths."
        )

    try:
        video_frequency, video_amplitude = compute_fft_spectrum(
            video,
            sampling_rate,
            window=fft_window,
            remove_mean=True,
        )
        imu_frequency, imu_amplitude = compute_fft_spectrum(
            imu,
            sampling_rate,
            window=fft_window,
            remove_mean=True,
        )
    except MetricsError as exc:
        raise PlottingError(str(exc)) from exc

    upper_limit = (
        sampling_rate / 2.0
        if max_frequency_hz is None
        else min(float(max_frequency_hz), sampling_rate / 2.0)
    )
    lower_limit = max(float(min_frequency_hz), 0.0)

    video_mask = (
        (video_frequency >= lower_limit)
        & (video_frequency <= upper_limit)
    )
    imu_mask = (
        (imu_frequency >= lower_limit)
        & (imu_frequency <= upper_limit)
    )

    figure, axis = plt.subplots(figsize=(12, 5))

    if logarithmic_y:
        axis.semilogy(
            video_frequency[video_mask],
            np.maximum(video_amplitude[video_mask], np.finfo(float).eps),
            linewidth=1.0,
            label=video_label,
        )
        axis.semilogy(
            imu_frequency[imu_mask],
            np.maximum(imu_amplitude[imu_mask], np.finfo(float).eps),
            linewidth=1.0,
            alpha=0.8,
            label=imu_label,
        )
    else:
        axis.plot(
            video_frequency[video_mask],
            video_amplitude[video_mask],
            linewidth=1.0,
            label=video_label,
        )
        axis.plot(
            imu_frequency[imu_mask],
            imu_amplitude[imu_mask],
            linewidth=1.0,
            alpha=0.8,
            label=imu_label,
        )

    axis.set_title(title)
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(lower_limit, upper_limit)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    return _save_and_close(figure, output_file, dpi)


def plot_cross_correlation(
    lags: Any,
    cross_correlation: Any,
    sampling_rate: float,
    output_file: str | Path,
    *,
    lag_samples: int | None = None,
    max_lag_seconds: float | None = None,
    normalize: bool = True,
    title: str = "Cross-correlation",
    dpi: int = 300,
) -> Path:
    lag_array = _as_finite_1d_array(lags, "lags")
    correlation = _as_finite_1d_array(
        cross_correlation,
        "cross_correlation",
    )

    if lag_array.size != correlation.size:
        raise PlottingError(
            "The lag vector and cross-correlation must have equal lengths."
        )

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise PlottingError("sampling_rate must be positive and finite.")

    lag_seconds = lag_array / float(sampling_rate)
    displayed_correlation = correlation.copy()

    if normalize:
        scale = float(np.max(np.abs(displayed_correlation)))
        if not np.isclose(scale, 0.0):
            displayed_correlation /= scale

    if max_lag_seconds is not None:
        if max_lag_seconds <= 0:
            raise PlottingError("max_lag_seconds must be positive.")
        mask = np.abs(lag_seconds) <= float(max_lag_seconds)
    else:
        mask = np.ones(lag_seconds.size, dtype=bool)

    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(
        lag_seconds[mask],
        displayed_correlation[mask],
        linewidth=1.0,
    )
    axis.axvline(0.0, linewidth=1.0, linestyle="--")

    if lag_samples is not None:
        best_lag_seconds = float(lag_samples) / float(sampling_rate)
        axis.axvline(
            best_lag_seconds,
            linewidth=1.2,
            linestyle=":",
            label=f"Best lag: {best_lag_seconds:.6f} s",
        )
        axis.legend()

    axis.set_title(title)
    axis.set_xlabel("Lag [s]")
    axis.set_ylabel(
        "Normalized correlation" if normalize else "Correlation"
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    return _save_and_close(figure, output_file, dpi)


def plot_validation_overview(
    time: Any,
    video_signal: Any,
    imu_signal: Any,
    sampling_rate: float,
    output_file: str | Path,
    *,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    fft_window: str = "hann",
    video_label: str = "Video",
    imu_label: str = "IMU",
    dpi: int = 300,
) -> Path:
    time_array, video, imu = _validate_time_domain_data(
        time,
        video_signal,
        imu_signal,
    )

    try:
        video_frequency, video_amplitude = compute_fft_spectrum(
            video,
            sampling_rate,
            window=fft_window,
            remove_mean=True,
        )
        imu_frequency, imu_amplitude = compute_fft_spectrum(
            imu,
            sampling_rate,
            window=fft_window,
            remove_mean=True,
        )
    except MetricsError as exc:
        raise PlottingError(str(exc)) from exc

    upper_limit = (
        sampling_rate / 2.0
        if max_frequency_hz is None
        else min(float(max_frequency_hz), sampling_rate / 2.0)
    )
    lower_limit = max(float(min_frequency_hz), 0.0)

    video_frequency_mask = (
        (video_frequency >= lower_limit)
        & (video_frequency <= upper_limit)
    )
    imu_frequency_mask = (
        (imu_frequency >= lower_limit)
        & (imu_frequency <= upper_limit)
    )

    figure, axes = plt.subplots(3, 1, figsize=(13, 12))

    axes[0].plot(
        time_array,
        video,
        linewidth=0.9,
        label=video_label,
    )
    axes[0].plot(
        time_array,
        imu,
        linewidth=0.9,
        alpha=0.8,
        label=imu_label,
    )
    axes[0].set_title("Synchronized time-domain signals")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    residual = video - imu
    axes[1].plot(time_array, residual, linewidth=0.9)
    axes[1].axhline(0.0, linewidth=1.0, linestyle="--")
    axes[1].set_title("Residual error")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Video - IMU")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        video_frequency[video_frequency_mask],
        video_amplitude[video_frequency_mask],
        linewidth=1.0,
        label=video_label,
    )
    axes[2].plot(
        imu_frequency[imu_frequency_mask],
        imu_amplitude[imu_frequency_mask],
        linewidth=1.0,
        alpha=0.8,
        label=imu_label,
    )
    axes[2].set_title("Frequency-domain comparison")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlim(lower_limit, upper_limit)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    return _save_and_close(figure, output_file, dpi)


def create_validation_plots(
    output_dir: str | Path,
    time: Any,
    video_signal: Any,
    imu_signal: Any,
    sampling_rate: float,
    *,
    lags: Any | None = None,
    cross_correlation: Any | None = None,
    lag_samples: int | None = None,
    video_label: str = "Video",
    imu_label: str = "IMU",
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
    fft_window: str = "hann",
    max_lag_seconds: float | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    paths["time_domain"] = plot_time_domain_comparison(
        time,
        video_signal,
        imu_signal,
        directory / "video_vs_imu_time_domain.png",
        video_label=video_label,
        imu_label=imu_label,
        dpi=dpi,
    )

    paths["residual"] = plot_residual_error(
        time,
        video_signal,
        imu_signal,
        directory / "video_vs_imu_residual.png",
        dpi=dpi,
    )

    paths["fft"] = plot_fft_comparison(
        video_signal,
        imu_signal,
        sampling_rate,
        directory / "video_vs_imu_fft.png",
        video_label=video_label,
        imu_label=imu_label,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        fft_window=fft_window,
        dpi=dpi,
    )

    paths["overview"] = plot_validation_overview(
        time,
        video_signal,
        imu_signal,
        sampling_rate,
        directory / "video_vs_imu_overview.png",
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        fft_window=fft_window,
        video_label=video_label,
        imu_label=imu_label,
        dpi=dpi,
    )

    if lags is not None and cross_correlation is not None:
        paths["cross_correlation"] = plot_cross_correlation(
            lags,
            cross_correlation,
            sampling_rate,
            directory / "video_vs_imu_cross_correlation.png",
            lag_samples=lag_samples,
            max_lag_seconds=max_lag_seconds,
            normalize=True,
            dpi=dpi,
        )

    return paths