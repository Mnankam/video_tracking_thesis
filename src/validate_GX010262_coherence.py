#!/usr/bin/env python3
"""
validate_GX010262_coherence.py
==============================

Reproducible frequency-domain external validation for GX010262.

This script compares video-derived acceleration signals with the three
linear IMU channels linx, liny, and linz using magnitude-squared coherence.

Background
----------
Previous diagnostics showed that:

- the IMU signals exhibit a strong spectral component close to 5 Hz,
- the video-derived acceleration also contains spectral components in this
  range, although its strongest peaks occur partly at lower frequencies,
- direct band-pass Pearson correlation remains weak and lag estimates can be
  ambiguous for narrow-band oscillatory signals.

Therefore, this script evaluates frequency-dependent coupling using
magnitude-squared coherence:

    C_xy(f) = |P_xy(f)|² / (P_xx(f) * P_yy(f))

where C_xy(f) ranges from 0 to 1.

A high coherence value indicates that the two signals share a consistent
frequency-domain relationship at a particular frequency. It does not imply
absolute amplitude agreement.

Important
---------
The video acceleration is expressed in px/s². No validated pixel-to-metre
calibration is used. Therefore this validation focuses on spectral
relationship and coherence, not absolute amplitude agreement.

Outputs
-------
comparison.csv
    Ranked comparison of all six video/IMU pairs.

summary.json
    Compact machine-readable summary.

validation_config.json
    Complete configuration used for the run.

validation.log
    Execution log.

plots/
    Coherence and PSD figures for all six signal pairs.

Author
------
Serge Kouomnankam
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, find_peaks, welch

from src.validation.imu_loader import load_imu_signal


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for the GX010262 coherence validation."""

    video_id: str

    video_path: Path
    imu_path: Path
    output_dir: Path

    video_time_column: str
    video_x_column: str
    video_y_column: str

    imu_axes: tuple[str, ...]

    target_sampling_rate_hz: float

    analysis_low_hz: float
    analysis_high_hz: float

    coherence_band_low_hz: float
    coherence_band_high_hz: float

    reference_frequency_hz: float
    reference_frequency_tolerance_hz: float

    welch_nperseg: int
    welch_overlap_fraction: float

    differentiation_method: str


CONFIG = ValidationConfig(
    video_id="GX010262",

    video_path=Path(
        "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
        "video_tracking_thesis/outputs/Lucas_Kanade_CPU_1/"
        "GX010262_lucas_kanade_timeseries.csv"
    ),

    imu_path=Path(
        "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
        "video_tracking_thesis/data/archives/measured_data/"
        "2022-04-16/2022-04-16_19.43.22/data.csv"
    ),

    output_dir=Path(
        "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
        "video_tracking_thesis/outputs/external_validation/"
        "GX010262_coherence1"
    ),

    video_time_column="time_seconds",
    video_x_column="lk_displacement_x",
    video_y_column="lk_displacement_y",

    imu_axes=(
        "linx",
        "liny",
        "linz",
    ),

    target_sampling_rate_hz=200.0,

    analysis_low_hz=2.0,
    analysis_high_hz=8.0,

    coherence_band_low_hz=4.5,
    coherence_band_high_hz=5.5,

    reference_frequency_hz=5.0,
    reference_frequency_tolerance_hz=0.15,

    welch_nperseg=2048,
    welch_overlap_fraction=0.5,

    differentiation_method="numpy.gradient",
)


# =============================================================================
# Logging
# =============================================================================


LOGGER = logging.getLogger(
    "validate_GX010262_coherence"
)


def configure_logging(
    output_dir: Path,
) -> None:
    """Configure console and file logging."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        output_dir
        / "validation.log"
    )

    LOGGER.setLevel(
        logging.INFO
    )

    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)-8s | "
        "%(message)s"
    )

    console = logging.StreamHandler(
        sys.stdout
    )

    console.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        console
    )

    LOGGER.addHandler(
        file_handler
    )


# =============================================================================
# Generic utilities
# =============================================================================


def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def to_jsonable(
    value: Any,
) -> Any:
    """Convert common Python/NumPy objects to JSON-compatible values."""

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        tuple,
    ):
        return [
            to_jsonable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            to_jsonable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): to_jsonable(
                item
            )
            for key, item in value.items()
        }

    return value


def write_json(
    path: Path,
    data: Any,
) -> None:
    """Write formatted UTF-8 JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            to_jsonable(
                data
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )

        handle.write(
            "\n"
        )


# =============================================================================
# Signal preparation
# =============================================================================


def prepare_uniform_signal(
    time: np.ndarray,
    values: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a signal onto a uniform common time axis.
    """

    time = np.asarray(
        time,
        dtype=float,
    ).reshape(-1)

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if time.size != values.size:
        raise ValueError(
            "Time and signal arrays have different lengths."
        )

    finite = (
        np.isfinite(time)
        & np.isfinite(values)
    )

    time = time[
        finite
    ]

    values = values[
        finite
    ]

    if time.size < 2:
        raise ValueError(
            "Signal contains fewer than two valid samples."
        )

    order = np.argsort(
        time,
        kind="stable",
    )

    time = time[
        order
    ]

    values = values[
        order
    ]

    unique_time, indices = np.unique(
        time,
        return_index=True,
    )

    time = unique_time
    values = values[
        indices
    ]

    if np.any(
        np.diff(time) <= 0
    ):
        raise ValueError(
            "Time axis is not strictly increasing."
        )

    start_time = max(
        start_time,
        float(time[0]),
    )

    end_time = min(
        end_time,
        float(time[-1]),
    )

    if end_time <= start_time:
        raise ValueError(
            "No valid common interpolation interval."
        )

    dt = (
        1.0 / fs
    )

    uniform_time = np.arange(
        start_time,
        end_time,
        dt,
        dtype=float,
    )

    if uniform_time.size < 2:
        raise ValueError(
            "Uniform time axis contains fewer than two samples."
        )

    uniform_values = np.interp(
        uniform_time,
        time,
        values,
    )

    return (
        uniform_time,
        uniform_values,
    )


def derive_video_acceleration(
    displacement: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Calculate video acceleration from displacement.

    Source displacement is measured in pixels, therefore the result has
    units px/s².
    """

    displacement = np.asarray(
        displacement,
        dtype=float,
    ).reshape(-1)

    dt = (
        1.0 / fs
    )

    velocity = np.gradient(
        displacement,
        dt,
    )

    acceleration = np.gradient(
        velocity,
        dt,
    )

    return np.asarray(
        acceleration,
        dtype=float,
    )


def remove_mean(
    values: np.ndarray,
) -> np.ndarray:
    """Remove DC component."""

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        values
        - np.mean(values)
    )


# =============================================================================
# Spectral analysis
# =============================================================================


def get_welch_parameters(
    signal_length: int,
    configured_nperseg: int,
    overlap_fraction: float,
) -> tuple[int, int]:
    """
    Determine safe Welch segment and overlap lengths.
    """

    if signal_length < 8:
        raise ValueError(
            "Signal is too short for Welch analysis."
        )

    nperseg = min(
        configured_nperseg,
        signal_length,
    )

    if nperseg < 8:
        raise ValueError(
            "Welch nperseg is too small."
        )

    noverlap = int(
        round(
            nperseg
            * overlap_fraction
        )
    )

    noverlap = min(
        noverlap,
        nperseg - 1,
    )

    return (
        nperseg,
        noverlap,
    )


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate power spectral density using Welch's method."""

    values = remove_mean(
        signal
    )

    frequencies, psd = welch(
        values,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    return (
        frequencies,
        psd,
    )


def compute_coherence_curve(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate magnitude-squared coherence.
    """

    a = remove_mean(
        signal_a
    )

    b = remove_mean(
        signal_b
    )

    frequencies, coherence_values = coherence(
        a,
        b,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )

    return (
        frequencies,
        coherence_values,
    )


def nearest_frequency_value(
    frequencies: np.ndarray,
    values: np.ndarray,
    target_frequency_hz: float,
) -> tuple[float, float]:
    """
    Return frequency bin nearest to a requested frequency.
    """

    index = int(
        np.argmin(
            np.abs(
                frequencies
                - target_frequency_hz
            )
        )
    )

    return (
        float(
            frequencies[index]
        ),
        float(
            values[index]
        ),
    )


def band_statistics(
    frequencies: np.ndarray,
    values: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> dict[str, float]:
    """
    Calculate maximum and mean value inside a frequency band.
    """

    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    if not np.any(mask):
        raise ValueError(
            f"No spectral bins inside {low_hz}-{high_hz} Hz."
        )

    band_frequencies = frequencies[
        mask
    ]

    band_values = values[
        mask
    ]

    maximum_index = int(
        np.argmax(
            band_values
        )
    )

    return {
        "band_max_value": float(
            band_values[
                maximum_index
            ]
        ),
        "band_max_frequency_hz": float(
            band_frequencies[
                maximum_index
            ]
        ),
        "band_mean_value": float(
            np.mean(
                band_values
            )
        ),
        "band_median_value": float(
            np.median(
                band_values
            )
        ),
    }


def dominant_frequency_in_band(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> tuple[float, float]:
    """
    Return strongest PSD component in a specified frequency band.
    """

    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    if not np.any(mask):
        raise ValueError(
            "No frequencies available inside requested analysis band."
        )

    band_f = frequencies[
        mask
    ]

    band_psd = psd[
        mask
    ]

    index = int(
        np.argmax(
            band_psd
        )
    )

    return (
        float(
            band_f[
                index
            ]
        ),
        float(
            band_psd[
                index
            ]
        ),
    )


def top_spectral_peaks(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
    limit: int = 10,
) -> list[dict[str, float]]:
    """
    Find strongest local PSD peaks in a given frequency range.
    """

    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    local_f = frequencies[
        mask
    ]

    local_psd = psd[
        mask
    ]

    if local_f.size < 3:
        return []

    peak_indices, _ = find_peaks(
        local_psd
    )

    if peak_indices.size == 0:
        return []

    order = peak_indices[
        np.argsort(
            local_psd[
                peak_indices
            ]
        )[::-1]
    ]

    output: list[
        dict[str, float]
    ] = []

    for index in order[
        :limit
    ]:
        output.append(
            {
                "frequency_hz": float(
                    local_f[
                        index
                    ]
                ),
                "psd": float(
                    local_psd[
                        index
                    ]
                ),
            }
        )

    return output


# =============================================================================
# Plotting
# =============================================================================


def create_pair_plot(
    output_path: Path,
    video_name: str,
    imu_axis: str,
    frequencies: np.ndarray,
    coherence_values: np.ndarray,
    video_psd_f: np.ndarray,
    video_psd: np.ndarray,
    imu_psd_f: np.ndarray,
    imu_psd: np.ndarray,
    config: ValidationConfig,
) -> None:
    """
    Create one coherence plot and associated PSD plot.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        frequencies,
        coherence_values,
    )

    plt.axvline(
        config.reference_frequency_hz,
        linestyle="--",
        linewidth=1,
        label="5 Hz reference",
    )

    plt.axvspan(
        config.coherence_band_low_hz,
        config.coherence_band_high_hz,
        alpha=0.15,
        label="4.5-5.5 Hz band",
    )

    plt.xlim(
        config.analysis_low_hz,
        config.analysis_high_hz,
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Frequency [Hz]"
    )

    plt.ylabel(
        "Magnitude-squared coherence"
    )

    plt.title(
        f"GX010262 coherence: {video_name} vs {imu_axis}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    psd_path = (
        output_path.parent
        / (
            output_path.stem
            + "_psd.png"
        )
    )

    plt.figure(
        figsize=(10, 5)
    )

    video_mask = (
        (video_psd_f >= config.analysis_low_hz)
        & (video_psd_f <= config.analysis_high_hz)
    )

    imu_mask = (
        (imu_psd_f >= config.analysis_low_hz)
        & (imu_psd_f <= config.analysis_high_hz)
    )

    video_values = (
        video_psd[
            video_mask
        ]
    )

    imu_values = (
        imu_psd[
            imu_mask
        ]
    )

    if np.max(
        video_values
    ) > 0:
        video_values = (
            video_values
            / np.max(
                video_values
            )
        )

    if np.max(
        imu_values
    ) > 0:
        imu_values = (
            imu_values
            / np.max(
                imu_values
            )
        )

    plt.plot(
        video_psd_f[
            video_mask
        ],
        video_values,
        label=video_name,
    )

    plt.plot(
        imu_psd_f[
            imu_mask
        ],
        imu_values,
        label=imu_axis,
    )

    plt.axvline(
        config.reference_frequency_hz,
        linestyle="--",
        linewidth=1,
    )

    plt.axvspan(
        config.coherence_band_low_hz,
        config.coherence_band_high_hz,
        alpha=0.15,
    )

    plt.xlabel(
        "Frequency [Hz]"
    )

    plt.ylabel(
        "Normalized PSD"
    )

    plt.title(
        f"GX010262 PSD comparison: {video_name} vs {imu_axis}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        psd_path,
        dpi=180,
    )

    plt.close()


# =============================================================================
# Input loading
# =============================================================================


def load_video(
    config: ValidationConfig,
) -> pd.DataFrame:
    """Load Lucas-Kanade video time series."""

    LOGGER.info(
        "Loading video time series: %s",
        config.video_path,
    )

    if not config.video_path.exists():
        raise FileNotFoundError(
            config.video_path
        )

    frame = pd.read_csv(
        config.video_path
    )

    required = {
        config.video_time_column,
        config.video_x_column,
        config.video_y_column,
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise KeyError(
            "Missing video columns: "
            f"{sorted(missing)}"
        )

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=list(
            required
        )
    ).copy()

    frame = frame.sort_values(
        config.video_time_column
    ).reset_index(
        drop=True
    )

    LOGGER.info(
        "Video samples: %d",
        len(
            frame
        ),
    )

    return frame


def load_all_imu_axes(
    config: ValidationConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Load linx, liny and linz."""

    output: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for axis in config.imu_axes:

        LOGGER.info(
            "Loading IMU axis: %s",
            axis,
        )

        result = load_imu_signal(
            config.imu_path,
            axis=axis,
        )

        output[
            axis
        ] = {
            "time": np.asarray(
                result[
                    "time"
                ],
                dtype=float,
            ),
            "signal": np.asarray(
                result[
                    "signal"
                ],
                dtype=float,
            ),
        }

        LOGGER.info(
            "%s: %d samples, duration %.6f s, fs %.6f Hz",
            axis,
            int(
                result[
                    "num_samples"
                ]
            ),
            float(
                result[
                    "duration"
                ]
            ),
            float(
                result[
                    "sampling_rate"
                ]
            ),
        )

    return output


# =============================================================================
# Pair evaluation
# =============================================================================


def evaluate_pair(
    video_name: str,
    video_signal: np.ndarray,
    imu_axis: str,
    imu_signal: np.ndarray,
    fs: float,
    config: ValidationConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """
    Evaluate one video/IMU pair.
    """

    signal_length = min(
        len(
            video_signal
        ),
        len(
            imu_signal
        ),
    )

    video_signal = np.asarray(
        video_signal[
            :signal_length
        ],
        dtype=float,
    )

    imu_signal = np.asarray(
        imu_signal[
            :signal_length
        ],
        dtype=float,
    )

    nperseg, noverlap = get_welch_parameters(
        signal_length=signal_length,
        configured_nperseg=config.welch_nperseg,
        overlap_fraction=config.welch_overlap_fraction,
    )

    coherence_f, coherence_values = compute_coherence_curve(
        signal_a=video_signal,
        signal_b=imu_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    video_psd_f, video_psd = compute_psd(
        signal=video_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    imu_psd_f, imu_psd = compute_psd(
        signal=imu_signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    band = band_statistics(
        frequencies=coherence_f,
        values=coherence_values,
        low_hz=config.coherence_band_low_hz,
        high_hz=config.coherence_band_high_hz,
    )

    reference_frequency_bin_hz, coherence_at_reference = (
        nearest_frequency_value(
            frequencies=coherence_f,
            values=coherence_values,
            target_frequency_hz=config.reference_frequency_hz,
        )
    )

    reference_mask = (
        coherence_f
        >= (
            config.reference_frequency_hz
            - config.reference_frequency_tolerance_hz
        )
    ) & (
        coherence_f
        <= (
            config.reference_frequency_hz
            + config.reference_frequency_tolerance_hz
        )
    )

    if np.any(
        reference_mask
    ):
        coherence_near_reference_mean = float(
            np.mean(
                coherence_values[
                    reference_mask
                ]
            )
        )

        coherence_near_reference_max = float(
            np.max(
                coherence_values[
                    reference_mask
                ]
            )
        )

    else:
        coherence_near_reference_mean = float(
            "nan"
        )

        coherence_near_reference_max = float(
            "nan"
        )

    video_dom_f, video_dom_psd = dominant_frequency_in_band(
        frequencies=video_psd_f,
        psd=video_psd,
        low_hz=config.analysis_low_hz,
        high_hz=config.analysis_high_hz,
    )

    imu_dom_f, imu_dom_psd = dominant_frequency_in_band(
        frequencies=imu_psd_f,
        psd=imu_psd,
        low_hz=config.analysis_low_hz,
        high_hz=config.analysis_high_hz,
    )

    frequency_difference_hz = float(
        abs(
            video_dom_f
            - imu_dom_f
        )
    )

    reference_frequency_difference_video_hz = float(
        abs(
            video_dom_f
            - config.reference_frequency_hz
        )
    )

    reference_frequency_difference_imu_hz = float(
        abs(
            imu_dom_f
            - config.reference_frequency_hz
        )
    )

    metrics = {
        "video_signal": video_name,
        "imu_axis": imu_axis,

        "samples": int(
            signal_length
        ),

        "duration_s": float(
            (
                signal_length
                - 1
            )
            / fs
        ),

        "sampling_rate_hz": float(
            fs
        ),

        "welch_nperseg": int(
            nperseg
        ),

        "welch_noverlap": int(
            noverlap
        ),

        "frequency_resolution_hz": float(
            coherence_f[1]
            - coherence_f[0]
        ),

        "coherence_at_5hz": float(
            coherence_at_reference
        ),

        "coherence_5hz_bin_hz": float(
            reference_frequency_bin_hz
        ),

        "coherence_near_5hz_mean": float(
            coherence_near_reference_mean
        ),

        "coherence_near_5hz_max": float(
            coherence_near_reference_max
        ),

        "band_max_coherence": float(
            band[
                "band_max_value"
            ]
        ),

        "band_max_coherence_frequency_hz": float(
            band[
                "band_max_frequency_hz"
            ]
        ),

        "band_mean_coherence": float(
            band[
                "band_mean_value"
            ]
        ),

        "band_median_coherence": float(
            band[
                "band_median_value"
            ]
        ),

        "video_dominant_frequency_2_8_hz": float(
            video_dom_f
        ),

        "imu_dominant_frequency_2_8_hz": float(
            imu_dom_f
        ),

        "dominant_frequency_difference_hz": float(
            frequency_difference_hz
        ),

        "video_distance_to_5hz": float(
            reference_frequency_difference_video_hz
        ),

        "imu_distance_to_5hz": float(
            reference_frequency_difference_imu_hz
        ),

        "video_dominant_psd": float(
            video_dom_psd
        ),

        "imu_dominant_psd": float(
            imu_dom_psd
        ),
    }

    diagnostics = {
        "coherence_frequency": coherence_f,
        "coherence": coherence_values,

        "video_psd_frequency": video_psd_f,
        "video_psd": video_psd,

        "imu_psd_frequency": imu_psd_f,
        "imu_psd": imu_psd,

        "video_top_peaks": top_spectral_peaks(
            video_psd_f,
            video_psd,
            config.analysis_low_hz,
            config.analysis_high_hz,
            limit=10,
        ),

        "imu_top_peaks": top_spectral_peaks(
            imu_psd_f,
            imu_psd,
            config.analysis_low_hz,
            config.analysis_high_hz,
            limit=10,
        ),
    }

    return (
        metrics,
        diagnostics,
    )


# =============================================================================
# Main validation
# =============================================================================


def run_validation(
    config: ValidationConfig,
) -> pd.DataFrame:
    """Execute complete GX010262 coherence validation."""

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = (
        config.output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "GX010262 coherence external validation"
    )

    LOGGER.info(
        "=" * 72
    )

    video = load_video(
        config
    )

    imu_data = load_all_imu_axes(
        config
    )

    video_time = video[
        config.video_time_column
    ].to_numpy(
        dtype=float
    )

    video_x = video[
        config.video_x_column
    ].to_numpy(
        dtype=float
    )

    video_y = video[
        config.video_y_column
    ].to_numpy(
        dtype=float
    )

    start_time = max(
        float(
            video_time[0]
        ),
        *[
            float(
                data[
                    "time"
                ][0]
            )
            for data
            in imu_data.values()
        ],
    )

    end_time = min(
        float(
            video_time[-1]
        ),
        *[
            float(
                data[
                    "time"
                ][-1]
            )
            for data
            in imu_data.values()
        ],
    )

    if end_time <= start_time:
        raise RuntimeError(
            "No temporal overlap between video and IMU."
        )

    LOGGER.info(
        "Common interval: %.6f to %.6f s",
        start_time,
        end_time,
    )

    LOGGER.info(
        "Common duration: %.6f s",
        end_time
        - start_time,
    )

    LOGGER.info(
        "Target sampling rate: %.3f Hz",
        config.target_sampling_rate_hz,
    )

    LOGGER.info(
        "Coherence band: %.3f to %.3f Hz",
        config.coherence_band_low_hz,
        config.coherence_band_high_hz,
    )

    LOGGER.info(
        "Reference frequency: %.3f Hz",
        config.reference_frequency_hz,
    )

    # -------------------------------------------------------------------------
    # Video interpolation
    # -------------------------------------------------------------------------

    common_time, video_x_uniform = prepare_uniform_signal(
        time=video_time,
        values=video_x,
        fs=config.target_sampling_rate_hz,
        start_time=start_time,
        end_time=end_time,
    )

    _, video_y_uniform = prepare_uniform_signal(
        time=video_time,
        values=video_y,
        fs=config.target_sampling_rate_hz,
        start_time=start_time,
        end_time=end_time,
    )

    # -------------------------------------------------------------------------
    # Video displacement -> acceleration
    # -------------------------------------------------------------------------

    video_signals = {
        "video_x_acc_px_s2": derive_video_acceleration(
            video_x_uniform,
            config.target_sampling_rate_hz,
        ),

        "video_y_acc_px_s2": derive_video_acceleration(
            video_y_uniform,
            config.target_sampling_rate_hz,
        ),
    }

    # -------------------------------------------------------------------------
    # IMU interpolation
    # -------------------------------------------------------------------------

    imu_signals: dict[
        str,
        np.ndarray,
    ] = {}

    for axis, data in imu_data.items():

        _, uniform_signal = prepare_uniform_signal(
            time=data[
                "time"
            ],
            values=data[
                "signal"
            ],
            fs=config.target_sampling_rate_hz,
            start_time=start_time,
            end_time=end_time,
        )

        imu_signals[
            axis
        ] = uniform_signal

    # -------------------------------------------------------------------------
    # Six pair evaluations
    # -------------------------------------------------------------------------

    rows: list[
        dict[str, Any]
    ] = []

    diagnostic_summary: dict[
        str,
        Any,
    ] = {}

    for video_name, video_signal in video_signals.items():

        for axis, imu_signal in imu_signals.items():

            LOGGER.info(
                "Evaluating coherence: %s vs %s",
                video_name,
                axis,
            )

            metrics, diagnostics = evaluate_pair(
                video_name=video_name,
                video_signal=video_signal,
                imu_axis=axis,
                imu_signal=imu_signal,
                fs=config.target_sampling_rate_hz,
                config=config,
            )

            rows.append(
                {
                    "video_id": config.video_id,
                    **metrics,
                }
            )

            key = (
                f"{video_name}__{axis}"
            )

            diagnostic_summary[
                key
            ] = {
                "video_top_peaks": diagnostics[
                    "video_top_peaks"
                ],
                "imu_top_peaks": diagnostics[
                    "imu_top_peaks"
                ],
            }

            plot_path = (
                plot_dir
                / (
                    f"{video_name}"
                    f"_vs_"
                    f"{axis}"
                    f"_coherence.png"
                )
            )

            create_pair_plot(
                output_path=plot_path,
                video_name=video_name,
                imu_axis=axis,
                frequencies=diagnostics[
                    "coherence_frequency"
                ],
                coherence_values=diagnostics[
                    "coherence"
                ],
                video_psd_f=diagnostics[
                    "video_psd_frequency"
                ],
                video_psd=diagnostics[
                    "video_psd"
                ],
                imu_psd_f=diagnostics[
                    "imu_psd_frequency"
                ],
                imu_psd=diagnostics[
                    "imu_psd"
                ],
                config=config,
            )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        by=[
            "coherence_at_5hz",
            "band_mean_coherence",
            "band_max_coherence",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    results.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                results
            )
            + 1,
        ),
    )

    # -------------------------------------------------------------------------
    # Save comparison
    # -------------------------------------------------------------------------

    comparison_path = (
        config.output_dir
        / "comparison.csv"
    )

    results.to_csv(
        comparison_path,
        index=False,
    )

    LOGGER.info(
        "Comparison saved: %s",
        comparison_path,
    )

    # -------------------------------------------------------------------------
    # Save configuration
    # -------------------------------------------------------------------------

    config_data = asdict(
        config
    )

    config_data[
        "created_utc"
    ] = utc_now_iso()

    config_data[
        "method"
    ] = {
        "coherence": (
            "Magnitude-squared coherence using scipy.signal.coherence."
        ),

        "video_acceleration": (
            "Second numerical derivative of Lucas-Kanade displacement "
            "using numpy.gradient."
        ),

        "resampling": (
            "Video and IMU interpolated to a common uniform 200-Hz grid."
        ),

        "psd": (
            "Welch power spectral density with Hann window."
        ),

        "ranking": (
            "Descending coherence at 5 Hz, then descending mean "
            "coherence in the 4.5-5.5 Hz band."
        ),

        "amplitude_note": (
            "Video acceleration remains in px/s². "
            "Absolute amplitude agreement is not evaluated."
        ),
    }

    write_json(
        config.output_dir
        / "validation_config.json",
        config_data,
    )

    # -------------------------------------------------------------------------
    # Save diagnostic peaks
    # -------------------------------------------------------------------------

    write_json(
        config.output_dir
        / "spectral_peaks.json",
        diagnostic_summary,
    )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    best = results.iloc[
        0
    ].to_dict()

    summary = {
        "created_utc": utc_now_iso(),

        "video_id": (
            config.video_id
        ),

        "inputs": {
            "video": str(
                config.video_path
            ),
            "imu": str(
                config.imu_path
            ),
        },

        "common_interval": {
            "start_time_s": float(
                start_time
            ),
            "end_time_s": float(
                end_time
            ),
            "duration_s": float(
                end_time
                - start_time
            ),
            "samples": int(
                len(
                    common_time
                )
            ),
        },

        "analysis": {
            "sampling_rate_hz": (
                config.target_sampling_rate_hz
            ),
            "analysis_low_hz": (
                config.analysis_low_hz
            ),
            "analysis_high_hz": (
                config.analysis_high_hz
            ),
            "coherence_band_low_hz": (
                config.coherence_band_low_hz
            ),
            "coherence_band_high_hz": (
                config.coherence_band_high_hz
            ),
            "reference_frequency_hz": (
                config.reference_frequency_hz
            ),
            "welch_nperseg": (
                config.welch_nperseg
            ),
            "welch_overlap_fraction": (
                config.welch_overlap_fraction
            ),
        },

        "number_of_comparisons": int(
            len(
                results
            )
        ),

        "best_pair": to_jsonable(
            best
        ),

        "interpretation_note": (
            "Coherence quantifies frequency-dependent linear coupling. "
            "It does not establish absolute amplitude agreement or "
            "prove physical causality."
        ),
    }

    write_json(
        config.output_dir
        / "summary.json",
        summary,
    )

    # -------------------------------------------------------------------------
    # Terminal output
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 120
    )

    print(
        "GX010262 COHERENCE VALIDATION RESULTS"
    )

    print(
        "=" * 120
    )

    display_columns = [
        "rank",
        "video_signal",
        "imu_axis",
        "coherence_at_5hz",
        "coherence_near_5hz_mean",
        "coherence_near_5hz_max",
        "band_max_coherence",
        "band_max_coherence_frequency_hz",
        "band_mean_coherence",
        "video_dominant_frequency_2_8_hz",
        "imu_dominant_frequency_2_8_hz",
        "dominant_frequency_difference_hz",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BEST PAIR"
    )

    print(
        "=" * 120
    )

    for key in [
        "video_signal",
        "imu_axis",
        "coherence_at_5hz",
        "coherence_5hz_bin_hz",
        "coherence_near_5hz_mean",
        "coherence_near_5hz_max",
        "band_max_coherence",
        "band_max_coherence_frequency_hz",
        "band_mean_coherence",
        "band_median_coherence",
        "video_dominant_frequency_2_8_hz",
        "imu_dominant_frequency_2_8_hz",
        "dominant_frequency_difference_hz",
        "frequency_resolution_hz",
    ]:
        print(
            f"{key}: "
            f"{best[key]}"
        )

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {config.output_dir}"
    )

    return results


# =============================================================================
# Entry point
# =============================================================================


def main() -> int:
    """Program entry point."""

    try:
        configure_logging(
            CONFIG.output_dir
        )

        run_validation(
            CONFIG
        )

        LOGGER.info(
            "Coherence validation completed successfully."
        )

        return 0

    except Exception:
        LOGGER.exception(
            "GX010262 coherence validation failed."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )