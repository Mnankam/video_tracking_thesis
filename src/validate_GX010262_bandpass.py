#!/usr/bin/env python3
"""
validate_GX010262_bandpass.py
=============================

Reproducible band-limited external validation for video GX010262.

This script compares video-derived motion information obtained from the
Lucas-Kanade time series with the three linear IMU axes linx, liny, and linz.

The validation focuses on the frequency band from 4.5 Hz to 5.5 Hz. This
frequency interval was selected because the IMU spectrum of GX010262 shows
a pronounced component close to 5 Hz, while the video-derived motion signal
also contains spectral components in this range.

The purpose of this script is not to perform an absolute amplitude validation.
The video signal is available in pixel coordinates and no experimentally
verified pixel-to-metre calibration is currently available. Therefore:

    - video acceleration is expressed in px/s²,
    - IMU values remain in their original measurement units,
    - z-score normalization is used for waveform comparison,
    - Pearson correlation is used for similarity,
    - cross-correlation is used for lag estimation,
    - RMS values are reported only within each original signal domain.

The script produces:

    comparison.csv
        Ranking of all six video/IMU signal combinations.

    validation_config.json
        Complete configuration and input paths used for the run.

    summary.json
        Compact machine-readable summary including the best pair.

    validation.log
        Execution log.

Method
------

1. Load the Lucas-Kanade frame-wise time series.
2. Load linx, liny, and linz using src.validation.imu_loader.
3. Restrict both measurement systems to their common duration.
4. Interpolate video and IMU signals to a common 200 Hz time grid.
5. Convert video displacement to acceleration using numerical derivatives.
6. Apply a zero-phase Butterworth band-pass filter from 4.5 Hz to 5.5 Hz.
7. Estimate temporal lag using cross-correlation within ±2 s.
8. Align signals according to the estimated lag.
9. Z-score normalize both signals.
10. Calculate Pearson correlation and normalized RMSE.
11. Rank all six combinations by absolute Pearson correlation.

Important
---------

This script is dataset-specific for GX010262. It intentionally keeps the
input files and main validation settings explicitly visible in the source
code so that the experiment can be reproduced exactly.

Author:
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

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, filtfilt

from src.validation.imu_loader import load_imu_signal


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration of the GX010262 band-pass validation.
    """

    video_id: str

    video_path: Path
    imu_path: Path
    output_dir: Path

    video_time_column: str
    video_x_column: str
    video_y_column: str

    imu_axes: tuple[str, ...]

    target_sampling_rate_hz: float

    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int

    maximum_lag_s: float

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
        "GX010262_bandpass_4p5_5p5_1"
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

    bandpass_low_hz=4.5,
    bandpass_high_hz=5.5,

    filter_order=4,

    maximum_lag_s=2.0,

    differentiation_method="numpy.gradient",
)


# =============================================================================
# Logging
# =============================================================================


LOGGER = logging.getLogger(
    "validate_GX010262_bandpass"
)


def configure_logging(
    output_dir: Path,
) -> None:
    """
    Configure terminal and file logging.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
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

    console_handler = (
        logging.StreamHandler(
            sys.stdout
        )
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = (
        logging.FileHandler(
            log_file,
            mode="w",
            encoding="utf-8",
        )
    )

    file_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        console_handler
    )

    LOGGER.addHandler(
        file_handler
    )


# =============================================================================
# Utility functions
# =============================================================================


def utc_now_iso() -> str:
    """
    Return current UTC timestamp in ISO-8601 format.
    """

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
    """
    Convert common Python/NumPy/Path objects into JSON-compatible values.
    """

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
    """
    Write JSON using UTF-8 and readable indentation.
    """

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
# Signal processing helpers
# =============================================================================


def validate_bandpass_parameters(
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> None:
    """
    Validate the band-pass filter configuration.
    """

    if fs <= 0:
        raise ValueError(
            "Sampling rate must be positive."
        )

    if low_hz <= 0:
        raise ValueError(
            "Lower band-pass frequency must be positive."
        )

    if high_hz <= low_hz:
        raise ValueError(
            "Upper band-pass frequency must be greater than lower frequency."
        )

    nyquist = (
        fs / 2.0
    )

    if high_hz >= nyquist:
        raise ValueError(
            f"Upper band-pass frequency ({high_hz} Hz) "
            f"must remain below Nyquist frequency ({nyquist} Hz)."
        )

    if order < 1:
        raise ValueError(
            "Filter order must be at least 1."
        )


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth band-pass filtering.
    """

    values = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    if values.size < 10:
        raise ValueError(
            "Signal contains too few samples for band-pass filtering."
        )

    validate_bandpass_parameters(
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        order=order,
    )

    nyquist = (
        0.5 * fs
    )

    normalized_low = (
        low_hz / nyquist
    )

    normalized_high = (
        high_hz / nyquist
    )

    b, a = butter(
        order,
        [
            normalized_low,
            normalized_high,
        ],
        btype="bandpass",
    )

    filtered = filtfilt(
        b,
        a,
        values,
    )

    return np.asarray(
        filtered,
        dtype=float,
    )


def prepare_uniform_signal(
    time: np.ndarray,
    values: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Interpolate a signal onto a common uniform time grid.
    """

    time = np.asarray(
        time,
        dtype=float,
    ).reshape(-1)

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if (
        time.size
        != values.size
    ):
        raise ValueError(
            "Time and signal arrays have different lengths."
        )

    finite = (
        np.isfinite(
            time
        )
        & np.isfinite(
            values
        )
    )

    time = (
        time[
            finite
        ]
    )

    values = (
        values[
            finite
        ]
    )

    if time.size < 2:
        raise ValueError(
            "Not enough valid samples after NaN/Inf removal."
        )

    order = np.argsort(
        time,
        kind="stable",
    )

    time = (
        time[
            order
        ]
    )

    values = (
        values[
            order
        ]
    )

    unique_time, indices = (
        np.unique(
            time,
            return_index=True,
        )
    )

    values = (
        values[
            indices
        ]
    )

    time = (
        unique_time
    )

    if np.any(
        np.diff(
            time
        )
        <= 0
    ):
        raise ValueError(
            "Time axis must be strictly increasing."
        )

    if start_time < time[0]:
        start_time = float(
            time[0]
        )

    if end_time > time[-1]:
        end_time = float(
            time[-1]
        )

    if end_time <= start_time:
        raise ValueError(
            "Invalid interpolation interval."
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
            "Uniform time grid contains fewer than two samples."
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


def zscore(
    values: np.ndarray,
) -> np.ndarray:
    """
    Standardize signal to zero mean and unit standard deviation.
    """

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    mean = float(
        np.mean(
            values
        )
    )

    std = float(
        np.std(
            values
        )
    )

    if (
        not np.isfinite(
            std
        )
        or std
        <= np.finfo(
            float
        ).eps
    ):
        raise ValueError(
            "Signal standard deviation is zero or invalid."
        )

    return (
        values - mean
    ) / std


def rms(
    values: np.ndarray,
) -> float:
    """
    Calculate root-mean-square amplitude.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def derive_video_acceleration(
    displacement: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Derive acceleration from displacement using two numerical gradients.

    The resulting unit is px/s² because the source displacement is measured
    in pixels.
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


def estimate_lag(
    reference: np.ndarray,
    target: np.ndarray,
    fs: float,
    max_lag_s: float,
) -> tuple[
    int,
    float,
    float,
]:
    """
    Estimate temporal lag by normalized cross-correlation.

    The search is restricted to ±max_lag_s.

    Returns
    -------
    lag_samples
        Optimal lag in samples.

    lag_seconds
        Optimal lag in seconds.

    normalized_peak
        Cross-correlation value normalized approximately by signal length.
    """

    reference_z = zscore(
        reference
    )

    target_z = zscore(
        target
    )

    correlation = correlate(
        target_z,
        reference_z,
        mode="full",
    )

    lags = np.arange(
        -len(
            reference_z
        )
        + 1,
        len(
            target_z
        ),
    )

    max_lag_samples = int(
        round(
            max_lag_s
            * fs
        )
    )

    valid = (
        np.abs(
            lags
        )
        <= max_lag_samples
    )

    correlation_valid = (
        correlation[
            valid
        ]
    )

    lags_valid = (
        lags[
            valid
        ]
    )

    if correlation_valid.size == 0:
        raise RuntimeError(
            "No lag candidates remain after lag restriction."
        )

    best_index = int(
        np.argmax(
            np.abs(
                correlation_valid
            )
        )
    )

    lag_samples = int(
        lags_valid[
            best_index
        ]
    )

    lag_seconds = float(
        lag_samples
        / fs
    )

    normalized_peak = float(
        correlation_valid[
            best_index
        ]
        / min(
            len(
                reference_z
            ),
            len(
                target_z
            ),
        )
    )

    return (
        lag_samples,
        lag_seconds,
        normalized_peak,
    )


def align_by_lag(
    reference: np.ndarray,
    target: np.ndarray,
    lag_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Shift two signals according to estimated lag and retain common samples.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    if lag_samples > 0:
        reference_aligned = (
            reference[
                :-lag_samples
            ]
        )

        target_aligned = (
            target[
                lag_samples:
            ]
        )

    elif lag_samples < 0:
        shift = (
            -lag_samples
        )

        reference_aligned = (
            reference[
                shift:
            ]
        )

        target_aligned = (
            target[
                :-shift
            ]
        )

    else:
        reference_aligned = (
            reference
        )

        target_aligned = (
            target
        )

    n = min(
        len(
            reference_aligned
        ),
        len(
            target_aligned
        ),
    )

    if n < 2:
        raise ValueError(
            "Too few samples remain after lag alignment."
        )

    return (
        reference_aligned[
            :n
        ],
        target_aligned[
            :n
        ],
    )


def calculate_pair_metrics(
    video_signal: np.ndarray,
    imu_signal: np.ndarray,
    fs: float,
    maximum_lag_s: float,
) -> dict[str, Any]:
    """
    Synchronize one video/IMU pair and calculate comparison metrics.
    """

    lag_samples, lag_s, correlation_peak = (
        estimate_lag(
            reference=video_signal,
            target=imu_signal,
            fs=fs,
            max_lag_s=maximum_lag_s,
        )
    )

    video_aligned, imu_aligned = (
        align_by_lag(
            reference=video_signal,
            target=imu_signal,
            lag_samples=lag_samples,
        )
    )

    video_z = zscore(
        video_aligned
    )

    imu_z = zscore(
        imu_aligned
    )

    pearson_r = float(
        np.corrcoef(
            video_z,
            imu_z,
        )[0, 1]
    )

    rmse_zscore = float(
        np.sqrt(
            np.mean(
                (
                    video_z
                    - imu_z
                )
                ** 2
            )
        )
    )

    mae_zscore = float(
        np.mean(
            np.abs(
                video_z
                - imu_z
            )
        )
    )

    return {
        "lag_samples": int(
            lag_samples
        ),
        "lag_s": float(
            lag_s
        ),
        "cross_correlation_peak": float(
            correlation_peak
        ),
        "pearson_r": float(
            pearson_r
        ),
        "abs_pearson_r": float(
            abs(
                pearson_r
            )
        ),
        "rmse_zscore": float(
            rmse_zscore
        ),
        "mae_zscore": float(
            mae_zscore
        ),
        "video_band_rms": rms(
            video_aligned
        ),
        "imu_band_rms": rms(
            imu_aligned
        ),
        "aligned_samples": int(
            len(
                video_aligned
            )
        ),
        "aligned_duration_s": float(
            (
                len(
                    video_aligned
                )
                - 1
            )
            / fs
        ),
    }


# =============================================================================
# Input loading
# =============================================================================


def load_video(
    config: ValidationConfig,
) -> pd.DataFrame:
    """
    Load and validate the Lucas-Kanade time-series file.
    """

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

    missing = (
        required
        .difference(
            frame.columns
        )
    )

    if missing:
        raise KeyError(
            "Missing required video columns: "
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

    if len(frame) < 2:
        raise RuntimeError(
            "Video time series contains fewer than two valid samples."
        )

    LOGGER.info(
        "Video samples: %d",
        len(
            frame
        ),
    )

    LOGGER.info(
        "Video time range: %.6f to %.6f s",
        frame[
            config.video_time_column
        ].iloc[0],
        frame[
            config.video_time_column
        ].iloc[-1],
    )

    return frame


def load_all_imu_axes(
    config: ValidationConfig,
) -> dict[
    str,
    dict[str, np.ndarray],
]:
    """
    Load all configured linear IMU axes.
    """

    if not config.imu_path.exists():
        raise FileNotFoundError(
            config.imu_path
        )

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

        time = np.asarray(
            result["time"],
            dtype=float,
        )

        signal = np.asarray(
            result["signal"],
            dtype=float,
        )

        output[
            axis
        ] = {
            "time": time,
            "signal": signal,
        }

        LOGGER.info(
            "%s: %d samples, %.6f s, effective fs %.6f Hz",
            axis,
            len(
                signal
            ),
            float(
                result["duration"]
            ),
            float(
                result["sampling_rate"]
            ),
        )

    return output


# =============================================================================
# Main validation
# =============================================================================


def run_validation(
    config: ValidationConfig,
) -> pd.DataFrame:
    """
    Execute the complete six-pair band-pass validation.
    """

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "GX010262 band-pass external validation"
    )

    LOGGER.info(
        "=" * 72
    )

    validate_bandpass_parameters(
        fs=config.target_sampling_rate_hz,
        low_hz=config.bandpass_low_hz,
        high_hz=config.bandpass_high_hz,
        order=config.filter_order,
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
        "Band-pass: %.3f to %.3f Hz",
        config.bandpass_low_hz,
        config.bandpass_high_hz,
    )

    LOGGER.info(
        "Filter order: %d",
        config.filter_order,
    )

    LOGGER.info(
        "Maximum lag search: ±%.3f s",
        config.maximum_lag_s,
    )

    # -------------------------------------------------------------------------
    # Video to uniform grid
    # -------------------------------------------------------------------------

    common_time, video_x_uniform = (
        prepare_uniform_signal(
            time=video_time,
            values=video_x,
            fs=config.target_sampling_rate_hz,
            start_time=start_time,
            end_time=end_time,
        )
    )

    _, video_y_uniform = (
        prepare_uniform_signal(
            time=video_time,
            values=video_y,
            fs=config.target_sampling_rate_hz,
            start_time=start_time,
            end_time=end_time,
        )
    )

    # -------------------------------------------------------------------------
    # Video displacement -> acceleration
    # -------------------------------------------------------------------------

    video_x_acceleration = (
        derive_video_acceleration(
            displacement=video_x_uniform,
            fs=config.target_sampling_rate_hz,
        )
    )

    video_y_acceleration = (
        derive_video_acceleration(
            displacement=video_y_uniform,
            fs=config.target_sampling_rate_hz,
        )
    )

    # -------------------------------------------------------------------------
    # Band-pass video acceleration
    # -------------------------------------------------------------------------

    video_signals = {
        "video_x_acc_px_s2": (
            bandpass_filter(
                signal=video_x_acceleration,
                fs=config.target_sampling_rate_hz,
                low_hz=config.bandpass_low_hz,
                high_hz=config.bandpass_high_hz,
                order=config.filter_order,
            )
        ),

        "video_y_acc_px_s2": (
            bandpass_filter(
                signal=video_y_acceleration,
                fs=config.target_sampling_rate_hz,
                low_hz=config.bandpass_low_hz,
                high_hz=config.bandpass_high_hz,
                order=config.filter_order,
            )
        ),
    }

    # -------------------------------------------------------------------------
    # IMU to same time grid + band-pass
    # -------------------------------------------------------------------------

    imu_signals: dict[
        str,
        np.ndarray,
    ] = {}

    for axis, data in imu_data.items():
        _, uniform_signal = (
            prepare_uniform_signal(
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
        )

        imu_signals[
            axis
        ] = bandpass_filter(
            signal=uniform_signal,
            fs=config.target_sampling_rate_hz,
            low_hz=config.bandpass_low_hz,
            high_hz=config.bandpass_high_hz,
            order=config.filter_order,
        )

    # -------------------------------------------------------------------------
    # Six pair comparisons
    # -------------------------------------------------------------------------

    rows: list[
        dict[str, Any]
    ] = []

    for video_name, video_signal in video_signals.items():

        for axis, imu_signal in imu_signals.items():

            LOGGER.info(
                "Comparing %s against %s",
                video_name,
                axis,
            )

            metrics = calculate_pair_metrics(
                video_signal=video_signal,
                imu_signal=imu_signal,
                fs=config.target_sampling_rate_hz,
                maximum_lag_s=config.maximum_lag_s,
            )

            row = {
                "video_id": config.video_id,
                "video_signal": video_name,
                "imu_axis": axis,
                **metrics,
            }

            rows.append(
                row
            )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        by=[
            "abs_pearson_r",
            "rmse_zscore",
        ],
        ascending=[
            False,
            True,
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
        "Comparison written to: %s",
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
        "method_description"
    ] = {
        "video_acceleration": (
            "Second numerical derivative of Lucas-Kanade "
            "displacement using numpy.gradient."
        ),
        "video_unit": "px/s^2",
        "imu_unit": (
            "Original sensor unit; no absolute amplitude "
            "comparison is performed."
        ),
        "filter": (
            "Zero-phase Butterworth band-pass using scipy.signal.filtfilt."
        ),
        "normalization": (
            "Z-score after temporal alignment."
        ),
        "lag_estimation": (
            "Absolute normalized cross-correlation within configured lag range."
        ),
        "ranking": (
            "Descending absolute Pearson correlation, "
            "then ascending z-score RMSE."
        ),
    }

    write_json(
        config.output_dir
        / "validation_config.json",
        config_data,
    )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    best = (
        results
        .iloc[0]
        .to_dict()
    )

    summary = {
        "created_utc": utc_now_iso(),
        "video_id": config.video_id,

        "input": {
            "video_path": str(
                config.video_path
            ),
            "imu_path": str(
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
            "uniform_samples": int(
                len(
                    common_time
                )
            ),
        },

        "processing": {
            "sampling_rate_hz": (
                config.target_sampling_rate_hz
            ),
            "bandpass_low_hz": (
                config.bandpass_low_hz
            ),
            "bandpass_high_hz": (
                config.bandpass_high_hz
            ),
            "filter_order": (
                config.filter_order
            ),
            "maximum_lag_s": (
                config.maximum_lag_s
            ),
            "differentiation_method": (
                config.differentiation_method
            ),
        },

        "number_of_comparisons": int(
            len(
                results
            )
        ),

        "best_pair": (
            to_jsonable(
                best
            )
        ),

        "interpretation_note": (
            "Ranking identifies the strongest band-limited "
            "waveform similarity. It does not establish absolute "
            "amplitude agreement because the video signal is "
            "expressed in px/s² and no validated pixel-to-metre "
            "calibration is applied."
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
        "=" * 100
    )

    print(
        "GX010262 BANDPASS VALIDATION RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        results[
            [
                "rank",
                "video_signal",
                "imu_axis",
                "lag_s",
                "pearson_r",
                "abs_pearson_r",
                "rmse_zscore",
                "mae_zscore",
                "video_band_rms",
                "imu_band_rms",
                "aligned_samples",
            ]
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
        "=" * 100
    )

    print(
        "BEST PAIR"
    )

    print(
        "=" * 100
    )

    for key in [
        "video_signal",
        "imu_axis",
        "lag_samples",
        "lag_s",
        "pearson_r",
        "abs_pearson_r",
        "rmse_zscore",
        "mae_zscore",
        "video_band_rms",
        "imu_band_rms",
        "aligned_samples",
        "aligned_duration_s",
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
    """
    Main program entry point.
    """

    try:
        configure_logging(
            CONFIG.output_dir
        )

        run_validation(
            CONFIG
        )

        LOGGER.info(
            "Validation completed successfully."
        )

        return 0

    except Exception:
        LOGGER.exception(
            "GX010262 band-pass validation failed."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )