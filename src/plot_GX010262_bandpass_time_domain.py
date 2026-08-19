#!/usr/bin/env python3
"""
plot_GX010262_bandpass_time_domain.py
=====================================

Create reproducible time-domain figures for the external validation of
GX010262.

The script visualizes the same signal pair and processing chain used in the
final reproducible band-pass validation:

    video_x_acc_px_s2  <->  IMU linx

Processing chain
----------------
1. Load the Lucas-Kanade displacement time series.
2. Load the confirmed GX010262 IMU measurement using the project IMU loader.
3. Restrict both modalities to their common temporal interval.
4. Interpolate both signals onto a common 200-Hz grid.
5. Derive video acceleration from horizontal Lucas-Kanade displacement.
6. Apply the same fourth-order 4.5-5.5-Hz Butterworth band-pass.
7. Apply the lag obtained from the final band-pass validation:
       1.875 s = 375 samples at 200 Hz.
8. Z-score normalize both aligned signals.
9. Calculate Pearson correlation for consistency checking.
10. Create:
       - a full-duration comparison plot,
       - a fixed-duration zoom plot.

Important
---------
The figure is intended to visualize the already performed external
validation. It does not define a new validation procedure and does not
re-estimate the lag.

The video acceleration remains in px/s² before normalization. Therefore,
the plotted normalized amplitudes are dimensionless.

Author
------
Serge Kouomnankam
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.validation.imu_loader import load_imu_signal


# =============================================================================
# Configuration
# =============================================================================

VIDEO_ID = "GX010262"

VIDEO_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/"
    "Lucas_Kanade_CPU_1/"
    "GX010262_lucas_kanade_timeseries.csv"
)

IMU_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/data/archives/measured_data/"
    "2022-04-16/2022-04-16_19.43.22/data.csv"
)

OUTPUT_DIR = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/"
    "external_validation/"
    "GX010262_bandpass_4p5_5p5/"
    "time_domain_plots"
)

FULL_OUTPUT_PNG = (
    OUTPUT_DIR
    / "GX010262_bandpass_time_domain_full.png"
)

ZOOM_OUTPUT_PNG = (
    OUTPUT_DIR
    / "GX010262_bandpass_time_domain_zoom.png"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "GX010262_bandpass_time_domain_summary.json"
)


# -----------------------------------------------------------------------------
# Signal definitions
# -----------------------------------------------------------------------------

VIDEO_TIME_COLUMN = "time_seconds"
VIDEO_DISPLACEMENT_COLUMN = "lk_displacement_x"

IMU_AXIS = "linx"

TARGET_FS = 200.0

BANDPASS_LOW_HZ = 4.5
BANDPASS_HIGH_HZ = 5.5
FILTER_ORDER = 4

# Final lag from validate_GX010262_bandpass.py:
#
# video_x_acc_px_s2 <-> linx
#
# lag_samples = 375
# lag_s       = 1.875
VALIDATION_LAG_S = 1.875


# -----------------------------------------------------------------------------
# Zoom figure
# -----------------------------------------------------------------------------
#
# This interval is used only for visualization.
# It has no influence on validation metrics.
#
# A fixed interval is stored explicitly so that the figure remains
# reproducible.
#
ZOOM_START_S = 10.0
ZOOM_DURATION_S = 10.0


# =============================================================================
# Helper functions
# =============================================================================


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.
    """

    values = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    nyquist = 0.5 * fs

    if not (
        0.0
        < low_hz
        < high_hz
        < nyquist
    ):
        raise ValueError(
            "Invalid band-pass frequencies: "
            f"{low_hz}-{high_hz} Hz "
            f"for fs={fs} Hz."
        )

    b, a = butter(
        order,
        [
            low_hz / nyquist,
            high_hz / nyquist,
        ],
        btype="bandpass",
    )

    return filtfilt(
        b,
        a,
        values,
    )


def zscore(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Standardize a signal to zero mean and unit standard deviation.
    """

    values = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    if (
        not np.isfinite(std)
        or std <= np.finfo(float).eps
    ):
        raise ValueError(
            "Signal standard deviation is zero or invalid."
        )

    return (
        values - mean
    ) / std


def prepare_uniform_signal(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Interpolate a signal onto a uniform time grid.
    """

    time = np.asarray(
        time,
        dtype=float,
    ).reshape(-1)

    signal = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    if time.size != signal.size:
        raise ValueError(
            "Time and signal vectors have different lengths."
        )

    finite = (
        np.isfinite(time)
        & np.isfinite(signal)
    )

    time = time[finite]
    signal = signal[finite]

    if time.size < 2:
        raise ValueError(
            "Signal contains fewer than two valid samples."
        )

    order = np.argsort(
        time,
        kind="stable",
    )

    time = time[order]
    signal = signal[order]

    unique_time, indices = np.unique(
        time,
        return_index=True,
    )

    time = unique_time
    signal = signal[indices]

    if np.any(
        np.diff(time) <= 0
    ):
        raise ValueError(
            "Time axis must be strictly increasing."
        )

    start_time = max(
        float(start_time),
        float(time[0]),
    )

    end_time = min(
        float(end_time),
        float(time[-1]),
    )

    if end_time <= start_time:
        raise ValueError(
            "No valid interpolation interval."
        )

    dt = 1.0 / fs

    uniform_time = np.arange(
        start_time,
        end_time,
        dt,
        dtype=float,
    )

    uniform_signal = np.interp(
        uniform_time,
        time,
        signal,
    )

    return (
        uniform_time,
        uniform_signal,
    )


def derive_video_acceleration(
    displacement: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Derive video acceleration from Lucas-Kanade displacement.

    Input
    -----
    displacement:
        Image displacement in pixels.

    Output
    ------
    acceleration:
        Video-derived acceleration in px/s².
    """

    displacement = np.asarray(
        displacement,
        dtype=float,
    )

    dt = 1.0 / fs

    velocity = np.gradient(
        displacement,
        dt,
    )

    acceleration = np.gradient(
        velocity,
        dt,
    )

    return acceleration


def apply_positive_lag(
    video_signal: np.ndarray,
    imu_signal: np.ndarray,
    time: np.ndarray,
    lag_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply the lag convention used in validate_GX010262_bandpass.py.

    For a positive lag:
        video[:-lag]
        imu[lag:]
    """

    if lag_samples < 0:
        raise ValueError(
            "This figure is configured for the final positive "
            "GX010262 lag."
        )

    if lag_samples == 0:
        n = min(
            len(video_signal),
            len(imu_signal),
            len(time),
        )

        return (
            time[:n],
            video_signal[:n],
            imu_signal[:n],
        )

    if lag_samples >= min(
        len(video_signal),
        len(imu_signal),
    ):
        raise ValueError(
            "Lag exceeds signal length."
        )

    video_aligned = (
        video_signal[:-lag_samples]
    )

    imu_aligned = (
        imu_signal[lag_samples:]
    )

    # The displayed time axis refers to the retained video samples.
    time_aligned = (
        time[:-lag_samples]
    )

    n = min(
        len(video_aligned),
        len(imu_aligned),
        len(time_aligned),
    )

    return (
        time_aligned[:n],
        video_aligned[:n],
        imu_aligned[:n],
    )


def save_plot(
    time: np.ndarray,
    video: np.ndarray,
    imu: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    """
    Save one normalized time-domain comparison.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 4.8),
    )

    ax.plot(
        time,
        video,
        linewidth=1.1,
        label="Video-derived acceleration",
    )

    ax.plot(
        time,
        imu,
        linewidth=1.1,
        label="IMU linx",
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Normalized amplitude [-]"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Load video data
# =============================================================================

if not VIDEO_CSV.exists():
    raise FileNotFoundError(
        VIDEO_CSV
    )

video_df = pd.read_csv(
    VIDEO_CSV
)

required_video_columns = {
    VIDEO_TIME_COLUMN,
    VIDEO_DISPLACEMENT_COLUMN,
}

missing_video_columns = (
    required_video_columns
    .difference(
        video_df.columns
    )
)

if missing_video_columns:
    raise KeyError(
        "Missing video columns: "
        f"{sorted(missing_video_columns)}"
    )


video_time = pd.to_numeric(
    video_df[
        VIDEO_TIME_COLUMN
    ],
    errors="coerce",
).to_numpy(
    dtype=float
)

video_displacement_x = pd.to_numeric(
    video_df[
        VIDEO_DISPLACEMENT_COLUMN
    ],
    errors="coerce",
).to_numpy(
    dtype=float
)

valid_video = (
    np.isfinite(video_time)
    & np.isfinite(video_displacement_x)
)

video_time = (
    video_time[
        valid_video
    ]
)

video_displacement_x = (
    video_displacement_x[
        valid_video
    ]
)


# =============================================================================
# Load IMU through the validated project loader
# =============================================================================

imu_result = load_imu_signal(
    IMU_CSV,
    axis=IMU_AXIS,
)

imu_time = np.asarray(
    imu_result["time"],
    dtype=float,
)

imu_linx = np.asarray(
    imu_result["signal"],
    dtype=float,
)


# =============================================================================
# Common physical time interval
# =============================================================================

common_start = max(
    float(video_time[0]),
    float(imu_time[0]),
)

common_end = min(
    float(video_time[-1]),
    float(imu_time[-1]),
)

if common_end <= common_start:
    raise RuntimeError(
        "Video and IMU data have no common time interval."
    )


# =============================================================================
# Resample video and IMU to common 200-Hz grid
# =============================================================================

common_time, video_displacement_uniform = (
    prepare_uniform_signal(
        time=video_time,
        signal=video_displacement_x,
        fs=TARGET_FS,
        start_time=common_start,
        end_time=common_end,
    )
)

_, imu_uniform = (
    prepare_uniform_signal(
        time=imu_time,
        signal=imu_linx,
        fs=TARGET_FS,
        start_time=common_start,
        end_time=common_end,
    )
)


# =============================================================================
# Video displacement -> video acceleration
# =============================================================================

video_acceleration = (
    derive_video_acceleration(
        displacement=video_displacement_uniform,
        fs=TARGET_FS,
    )
)


# =============================================================================
# Same band-pass used in external validation
# =============================================================================

video_filtered = bandpass_filter(
    signal=video_acceleration,
    fs=TARGET_FS,
    low_hz=BANDPASS_LOW_HZ,
    high_hz=BANDPASS_HIGH_HZ,
    order=FILTER_ORDER,
)

imu_filtered = bandpass_filter(
    signal=imu_uniform,
    fs=TARGET_FS,
    low_hz=BANDPASS_LOW_HZ,
    high_hz=BANDPASS_HIGH_HZ,
    order=FILTER_ORDER,
)


# =============================================================================
# Apply final validation lag
# =============================================================================

lag_samples = int(
    round(
        VALIDATION_LAG_S
        * TARGET_FS
    )
)

time_aligned, video_aligned, imu_aligned = (
    apply_positive_lag(
        video_signal=video_filtered,
        imu_signal=imu_filtered,
        time=common_time,
        lag_samples=lag_samples,
    )
)


# =============================================================================
# Normalize exactly for waveform comparison
# =============================================================================

video_normalized = zscore(
    video_aligned
)

imu_normalized = zscore(
    imu_aligned
)


# =============================================================================
# Consistency metrics
# =============================================================================

pearson_r = float(
    np.corrcoef(
        video_normalized,
        imu_normalized,
    )[0, 1]
)

abs_pearson_r = abs(
    pearson_r
)

rmse_zscore = float(
    np.sqrt(
        np.mean(
            (
                video_normalized
                - imu_normalized
            )
            ** 2
        )
    )
)

aligned_samples = len(
    time_aligned
)

aligned_duration_s = (
    (aligned_samples - 1)
    / TARGET_FS
)


# =============================================================================
# Full-duration figure
# =============================================================================

save_plot(
    time=time_aligned,
    video=video_normalized,
    imu=imu_normalized,
    output_path=FULL_OUTPUT_PNG,
    title=(
        "GX010262 band-limited video--IMU comparison "
        "(4.5--5.5 Hz)"
    ),
)


# =============================================================================
# Fixed zoom figure
# =============================================================================

zoom_end_s = (
    ZOOM_START_S
    + ZOOM_DURATION_S
)

zoom_mask = (
    (time_aligned >= ZOOM_START_S)
    & (time_aligned <= zoom_end_s)
)

if not np.any(
    zoom_mask
):
    raise RuntimeError(
        "Configured zoom interval does not overlap the aligned data."
    )

save_plot(
    time=time_aligned[
        zoom_mask
    ],
    video=video_normalized[
        zoom_mask
    ],
    imu=imu_normalized[
        zoom_mask
    ],
    output_path=ZOOM_OUTPUT_PNG,
    title=(
        "GX010262 band-limited video--IMU comparison "
        f"({ZOOM_START_S:.0f}--{zoom_end_s:.0f} s)"
    ),
)


# =============================================================================
# Save machine-readable figure metadata
# =============================================================================

summary = {
    "video_id": VIDEO_ID,

    "video_csv": str(
        VIDEO_CSV
    ),

    "imu_csv": str(
        IMU_CSV
    ),

    "video_signal": (
        "video_x_acc_px_s2"
    ),

    "imu_axis": (
        IMU_AXIS
    ),

    "target_sampling_rate_hz": (
        TARGET_FS
    ),

    "bandpass_low_hz": (
        BANDPASS_LOW_HZ
    ),

    "bandpass_high_hz": (
        BANDPASS_HIGH_HZ
    ),

    "filter_order": (
        FILTER_ORDER
    ),

    "lag_s": (
        VALIDATION_LAG_S
    ),

    "lag_samples": (
        lag_samples
    ),

    "common_start_s": (
        common_start
    ),

    "common_end_s": (
        common_end
    ),

    "aligned_samples": (
        aligned_samples
    ),

    "aligned_duration_s": (
        aligned_duration_s
    ),

    "pearson_r": (
        pearson_r
    ),

    "abs_pearson_r": (
        abs_pearson_r
    ),

    "rmse_zscore": (
        rmse_zscore
    ),

    "zoom_start_s": (
        ZOOM_START_S
    ),

    "zoom_end_s": (
        zoom_end_s
    ),

    "full_figure": str(
        FULL_OUTPUT_PNG
    ),

    "zoom_figure": str(
        ZOOM_OUTPUT_PNG
    ),
}

SUMMARY_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )

    handle.write(
        "\n"
    )


# =============================================================================
# Terminal summary
# =============================================================================

print(
    "=" * 78
)

print(
    "GX010262 BAND-PASS TIME-DOMAIN FIGURE"
)

print(
    "=" * 78
)

print(
    f"Video signal        : video_x_acc_px_s2"
)

print(
    f"IMU signal          : {IMU_AXIS}"
)

print(
    f"Video samples input : {len(video_time)}"
)

print(
    f"IMU samples input   : {len(imu_time)}"
)

print(
    f"Common interval     : "
    f"{common_start:.6f} - {common_end:.6f} s"
)

print(
    f"Target rate         : "
    f"{TARGET_FS:.3f} Hz"
)

print(
    f"Band-pass           : "
    f"{BANDPASS_LOW_HZ:.3f} - "
    f"{BANDPASS_HIGH_HZ:.3f} Hz"
)

print(
    f"Filter order        : "
    f"{FILTER_ORDER}"
)

print(
    f"Applied lag         : "
    f"{VALIDATION_LAG_S:.6f} s"
)

print(
    f"Lag samples         : "
    f"{lag_samples}"
)

print(
    f"Aligned samples     : "
    f"{aligned_samples}"
)

print(
    f"Aligned duration    : "
    f"{aligned_duration_s:.6f} s"
)

print(
    f"Pearson r           : "
    f"{pearson_r:.6f}"
)

print(
    f"|Pearson r|         : "
    f"{abs_pearson_r:.6f}"
)

print(
    f"RMSE z-score        : "
    f"{rmse_zscore:.6f}"
)

print(
    f"Full figure         : "
    f"{FULL_OUTPUT_PNG}"
)

print(
    f"Zoom figure         : "
    f"{ZOOM_OUTPUT_PNG}"
)

print(
    f"Summary             : "
    f"{SUMMARY_JSON}"
)

print(
    "=" * 78
)