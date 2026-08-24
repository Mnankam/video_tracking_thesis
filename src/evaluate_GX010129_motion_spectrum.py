#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/"
    "Internal_Validation_GX010129/"
    "GX010129_lucas_kanade.csv"
)

OUTPUT_DIR = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/"
    "Internal_Validation_GX010129/"
    "motion_spectrum"
)

FPS = 200.0

# Main motion direction for this evaluation.
# Change to "dy" only if your inner-pipe motion was explicitly defined
# in the vertical image direction.
MOTION_COLUMN = "dx"

DOWNSAMPLING_FACTORS = [1, 2, 4, 5, 10]


# =============================================================================
# Load point-wise Lucas-Kanade results
# =============================================================================

df = pd.read_csv(INPUT_CSV)

required = {
    "frame",
    "time_seconds",
    "point_id",
    MOTION_COLUMN,
    "tracking_status",
}

missing = required.difference(df.columns)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

for column in required:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.dropna(
    subset=[
        "frame",
        "time_seconds",
        MOTION_COLUMN,
        "tracking_status",
    ]
).copy()

df["frame"] = df["frame"].astype(int)
df["tracking_status"] = df["tracking_status"].astype(int)

valid = df[
    df["tracking_status"] == 1
].copy()

if valid.empty:
    raise RuntimeError(
        "No valid Lucas-Kanade observations found."
    )


# =============================================================================
# Robust frame-wise motion signal
# =============================================================================

signal_df = (
    valid.groupby(
        ["frame", "time_seconds"],
        as_index=False,
    )
    .agg(
        motion=(
            MOTION_COLUMN,
            "median",
        ),
        valid_points=(
            "point_id",
            "nunique",
        ),
    )
    .sort_values(
        ["frame", "time_seconds"]
    )
    .reset_index(drop=True)
)

# Require at least two valid LK points, consistent with the extraction
# criterion used in the internal validation.
signal_df = signal_df[
    signal_df["valid_points"] >= 2
].copy()

if signal_df.empty:
    raise RuntimeError(
        "No frame satisfies the minimum of two valid tracking points."
    )

time = signal_df[
    "time_seconds"
].to_numpy(float)

motion = signal_df[
    "motion"
].to_numpy(float)


# =============================================================================
# FFT helper
# =============================================================================

def evaluate_spectrum(
    values: np.ndarray,
    fs: float,
) -> dict:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values - np.mean(values)

    n = len(values)

    if n < 4:
        raise ValueError(
            "Too few samples for spectral analysis."
        )

    window = np.hanning(n)

    spectrum = np.abs(
        np.fft.rfft(
            values * window
        )
    )

    frequency = np.fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    # Ignore zero frequency.
    positive = frequency > 0

    freq_pos = frequency[positive]
    spec_pos = spectrum[positive]

    if len(spec_pos) == 0:
        raise RuntimeError(
            "No positive-frequency bins available."
        )

    peaks, _ = find_peaks(
        spec_pos
    )

    if len(peaks) > 0:
        best_peak = peaks[
            np.argmax(
                spec_pos[peaks]
            )
        ]
    else:
        best_peak = int(
            np.argmax(
                spec_pos
            )
        )

    dominant_frequency = float(
        freq_pos[best_peak]
    )

    dominant_amplitude = float(
        spec_pos[best_peak]
    )

    return {
        "frequency": frequency,
        "spectrum": spectrum,
        "samples": int(n),
        "sampling_rate_hz": float(fs),
        "frequency_resolution_hz": float(
            fs / n
        ),
        "nyquist_hz": float(
            fs / 2.0
        ),
        "dominant_frequency_hz": (
            dominant_frequency
        ),
        "dominant_amplitude": (
            dominant_amplitude
        ),
    }


# =============================================================================
# Full-resolution spectrum
# =============================================================================

reference = evaluate_spectrum(
    motion,
    FPS,
)

reference_frequency = (
    reference[
        "dominant_frequency_hz"
    ]
)


# =============================================================================
# Downsampling evaluation
# =============================================================================

rows = []

spectra = {}

for q in DOWNSAMPLING_FACTORS:

    values_q = motion[::q]

    fs_q = FPS / q

    result = evaluate_spectrum(
        values_q,
        fs_q,
    )

    difference = abs(
        result[
            "dominant_frequency_hz"
        ]
        - reference_frequency
    )

    rows.append(
        {
            "downsampling_factor": q,
            "samples": result[
                "samples"
            ],
            "sampling_rate_hz": (
                result[
                    "sampling_rate_hz"
                ]
            ),
            "nyquist_hz": (
                result[
                    "nyquist_hz"
                ]
            ),
            "frequency_resolution_hz": (
                result[
                    "frequency_resolution_hz"
                ]
            ),
            "dominant_frequency_hz": (
                result[
                    "dominant_frequency_hz"
                ]
            ),
            "dominant_frequency_difference_hz": (
                difference
            ),
        }
    )

    spectra[q] = result


results = pd.DataFrame(
    rows
)


# =============================================================================
# Outputs
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

signal_df.to_csv(
    OUTPUT_DIR
    / "GX010129_framewise_motion.csv",
    index=False,
)

results.to_csv(
    OUTPUT_DIR
    / "GX010129_downsampling_results.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Full spectrum
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 4.8)
)

ax.plot(
    reference["frequency"],
    reference["spectrum"],
)

ax.set_xlim(
    0,
    min(
        30,
        reference["nyquist_hz"],
    ),
)

ax.set_xlabel(
    "Frequency [Hz]"
)

ax.set_ylabel(
    "Spectral magnitude"
)

ax.set_title(
    "GX010129 Lucas-Kanade motion spectrum"
)

ax.grid(
    True,
    alpha=0.3,
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "GX010129_motion_fft.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# -----------------------------------------------------------------------------
# Downsampling comparison
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 5)
)

for q, result in spectra.items():

    ax.plot(
        result["frequency"],
        result["spectrum"],
        label=f"q={q}, fs={FPS/q:.0f} Hz",
    )

ax.set_xlim(
    0,
    30,
)

ax.set_xlabel(
    "Frequency [Hz]"
)

ax.set_ylabel(
    "Spectral magnitude"
)

ax.set_title(
    "GX010129 spectral preservation under temporal downsampling"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "GX010129_downsampling_spectra.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# JSON summary
# =============================================================================

summary = {
    "video_id": "GX010129",
    "motion_column": MOTION_COLUMN,
    "native_sampling_rate_hz": FPS,
    "framewise_samples": int(
        len(signal_df)
    ),
    "reference_frequency_resolution_hz": (
        reference[
            "frequency_resolution_hz"
        ]
    ),
    "reference_nyquist_hz": (
        reference[
            "nyquist_hz"
        ]
    ),
    "reference_dominant_frequency_hz": (
        reference_frequency
    ),
    "downsampling": (
        results.to_dict(
            orient="records"
        )
    ),
}

with (
    OUTPUT_DIR
    / "GX010129_motion_spectrum_summary.json"
).open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        summary,
        handle,
        indent=2,
    )

    handle.write("\n")


# =============================================================================
# Terminal output
# =============================================================================

print("=" * 100)
print("GX010129 MOTION SPECTRUM")
print("=" * 100)

print(
    f"Motion column                : {MOTION_COLUMN}"
)

print(
    f"Frame-wise samples           : {len(signal_df)}"
)

print(
    f"Native sampling rate [Hz]    : {FPS:.6f}"
)

print(
    "Frequency resolution [Hz]   : "
    f"{reference['frequency_resolution_hz']:.9f}"
)

print(
    "Nyquist frequency [Hz]      : "
    f"{reference['nyquist_hz']:.6f}"
)

print(
    "Dominant frequency [Hz]     : "
    f"{reference_frequency:.9f}"
)

print()
print("=" * 100)
print("DOWNSAMPLING")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}",
    )
)

print()
print(
    f"Output directory: {OUTPUT_DIR}"
)