#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


# =============================================================================
# Configuration
# =============================================================================

VIDEO_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/"
    "Lucas_kanade_CPU_1/"
    "GX010262_lucas_kanade_timeseries.csv"
)

IMU_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/data/archives/measured_data/"
    "2022-04-16/2022-04-16_19.43.22/data.csv"
)

OUTPUT_PNG = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/"
    "external_validation/"
    "gx010262_time_domain_comparison.png"
)

VIDEO_FS = 200.0

# Validation band already used in the external validation
F_LOW = 4.5
F_HIGH = 5.5

# Lag obtained from the external validation
LAG_SECONDS = 1.875

# Representative time window for the thesis figure
WINDOW_START = 10.0
WINDOW_DURATION = 10.0


# =============================================================================
# Helpers
# =============================================================================

def butter_bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """
    Apply a zero-phase Butterworth band-pass filter.
    """

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band",
    )

    return filtfilt(
        b,
        a,
        signal,
    )


def normalize_signal(signal):
    """
    Standardize a signal to zero mean and unit standard deviation.
    """

    signal = np.asarray(signal, dtype=float)

    mean = np.nanmean(signal)
    std = np.nanstd(signal)

    if std == 0 or not np.isfinite(std):
        return signal - mean

    return (signal - mean) / std


# =============================================================================
# Load video signal
# =============================================================================

video_df = pd.read_csv(VIDEO_CSV)

required_video_columns = {
    "lk_displacement_x",
}

missing_video = required_video_columns.difference(video_df.columns)

if missing_video:
    raise KeyError(
        f"Missing video columns: {sorted(missing_video)}"
    )

video_x = video_df["lk_displacement_x"].to_numpy(dtype=float)

# Lucas-Kanade displacement starts between frame 0 and frame 1,
# therefore the first temporal sample corresponds to 1 / 200 s.
video_time = (
    np.arange(1, len(video_x) + 1, dtype=float)
    / VIDEO_FS
)

# Convert displacement to acceleration.
video_velocity = np.gradient(
    video_x,
    1.0 / VIDEO_FS,
)

video_acceleration = np.gradient(
    video_velocity,
    1.0 / VIDEO_FS,
)


# =============================================================================
# Load IMU signal
# =============================================================================

imu_df = pd.read_csv(IMU_CSV)

required_imu_columns = {
    "linx",
}

missing_imu = required_imu_columns.difference(imu_df.columns)

if missing_imu:
    raise KeyError(
        f"Missing IMU columns: {sorted(missing_imu)}"
    )

imu_x = imu_df["linx"].to_numpy(dtype=float)


# =============================================================================
# Determine IMU time axis
# =============================================================================

# Adjust this block if your IMU CSV already contains an explicit time column.
possible_time_columns = [
    "time_s",
    "time",
    "timestamp",
    "t",
]

imu_time_column = None

for column in possible_time_columns:
    if column in imu_df.columns:
        imu_time_column = column
        break

if imu_time_column is None:
    raise KeyError(
        "No IMU time column found. "
        f"Available columns: {list(imu_df.columns)}"
    )

imu_time_raw = imu_df[imu_time_column].to_numpy()

# If the time column is already numeric, interpret it as seconds.
if np.issubdtype(imu_time_raw.dtype, np.number):
    imu_time = imu_time_raw.astype(float)
    imu_time = imu_time - imu_time[0]

else:
    # Otherwise interpret it as timestamps.
    imu_datetime = pd.to_datetime(
        imu_time_raw,
        errors="coerce",
    )

    if imu_datetime.isna().all():
        raise ValueError(
            f"Could not interpret IMU time column "
            f"'{imu_time_column}'."
        )

    imu_time = (
        imu_datetime - imu_datetime.iloc[0]
    ).dt.total_seconds().to_numpy()


# =============================================================================
# Restrict to common time interval
# =============================================================================

common_start = max(
    video_time[0],
    imu_time[0],
)

common_end = min(
    video_time[-1],
    imu_time[-1],
)

if common_end <= common_start:
    raise RuntimeError(
        "Video and IMU signals have no common time interval."
    )

common_time = np.arange(
    common_start,
    common_end,
    1.0 / VIDEO_FS,
)


# =============================================================================
# Resample both signals onto the same 200 Hz grid
# =============================================================================

video_acc_common = np.interp(
    common_time,
    video_time,
    video_acceleration,
)

imu_x_common = np.interp(
    common_time,
    imu_time,
    imu_x,
)


# =============================================================================
# Band-pass filtering
# =============================================================================

video_filtered = butter_bandpass_filter(
    video_acc_common,
    VIDEO_FS,
    F_LOW,
    F_HIGH,
    order=4,
)

imu_filtered = butter_bandpass_filter(
    imu_x_common,
    VIDEO_FS,
    F_LOW,
    F_HIGH,
    order=4,
)


# =============================================================================
# Apply temporal alignment
# =============================================================================

lag_samples = int(
    round(LAG_SECONDS * VIDEO_FS)
)

if lag_samples > 0:
    video_aligned = video_filtered[:-lag_samples]
    imu_aligned = imu_filtered[lag_samples:]
    time_aligned = common_time[:-lag_samples]

elif lag_samples < 0:
    shift = abs(lag_samples)

    video_aligned = video_filtered[shift:]
    imu_aligned = imu_filtered[:-shift]
    time_aligned = common_time[shift:]

else:
    video_aligned = video_filtered
    imu_aligned = imu_filtered
    time_aligned = common_time


# =============================================================================
# Normalize for visual comparison
# =============================================================================

video_norm = normalize_signal(video_aligned)
imu_norm = normalize_signal(imu_aligned)


# =============================================================================
# Select representative time window
# =============================================================================

window_end = WINDOW_START + WINDOW_DURATION

window_mask = (
    (time_aligned >= WINDOW_START)
    & (time_aligned <= window_end)
)

if not np.any(window_mask):
    raise RuntimeError(
        "Selected time window does not overlap with aligned data."
    )

plot_time = time_aligned[window_mask]
plot_video = video_norm[window_mask]
plot_imu = imu_norm[window_mask]


# =============================================================================
# Plot
# =============================================================================

OUTPUT_PNG.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig, ax = plt.subplots(
    figsize=(10, 4.8),
)

ax.plot(
    plot_time,
    plot_video,
    label="Video-derived acceleration",
    linewidth=1.2,
)

ax.plot(
    plot_time,
    plot_imu,
    label="IMU linx",
    linewidth=1.2,
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized amplitude")

ax.set_title(
    "GX010262 synchronized band-limited signal comparison"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Summary
# =============================================================================

print("=" * 72)
print("TIME-DOMAIN COMPARISON CREATED")
print("=" * 72)
print(f"Video CSV       : {VIDEO_CSV}")
print(f"IMU CSV         : {IMU_CSV}")
print(f"Common rate     : {VIDEO_FS:.1f} Hz")
print(f"Band-pass       : {F_LOW:.1f}-{F_HIGH:.1f} Hz")
print(f"Applied lag     : {LAG_SECONDS:.3f} s")
print(f"Lag samples     : {lag_samples}")
print(
    f"Plot window     : "
    f"{WINDOW_START:.2f}-{window_end:.2f} s"
)
print(f"Output          : {OUTPUT_PNG}")
print("=" * 72)