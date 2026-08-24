#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Output directory
# =============================================================================

OUTPUT_DIR = Path(
    "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
    "video_tracking_thesis/outputs/Internal_Validation_GX010129/thesis_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Benchmark data
# =============================================================================

data = pd.DataFrame(
    {
        "Configuration": [
            "OpenCV\nsegmentation",
            "Lucas--\nKanade",
            "Farneback",
            "Detectron2\nGPU",
        ],
        "Throughput_fps": [
            76.00,
            12.64,
            11.98,
            25.64,
        ],
    }
)

# Calculate processing time directly from throughput.
data["Time_per_frame_s"] = (
    1.0 / data["Throughput_fps"]
)


# =============================================================================
# Throughput plot
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.5, 4.5)
)

bars = ax.bar(
    data["Configuration"],
    data["Throughput_fps"],
)

ax.set_ylabel("Throughput [fps]")
ax.set_xlabel("Processing configuration")
ax.set_title("Measured processing throughput")

ax.grid(
    axis="y",
    alpha=0.3,
)

# Add numerical values above bars.
for bar, value in zip(
    bars,
    data["Throughput_fps"],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()

fps_output = (
    OUTPUT_DIR /
    "performance_fps_comparison.png"
)

fig.savefig(
    fps_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Processing-time plot
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.5, 4.5)
)

bars = ax.bar(
    data["Configuration"],
    data["Time_per_frame_s"],
)

ax.set_ylabel("Mean processing time per frame [s]")
ax.set_xlabel("Processing configuration")
ax.set_title("Measured processing time per frame")

ax.grid(
    axis="y",
    alpha=0.3,
)

# Add numerical values above bars.
for bar, value in zip(
    bars,
    data["Time_per_frame_s"],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()

time_output = (
    OUTPUT_DIR /
    "processing_time_comparison.png"
)

fig.savefig(
    time_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Console output
# =============================================================================

print()
print("Benchmark data used for the plots:")
print()

print(
    data.to_string(
        index=False,
        formatters={
            "Throughput_fps": lambda x: f"{x:.2f}",
            "Time_per_frame_s": lambda x: f"{x:.4f}",
        },
    )
)

print()
print(f"FPS plot  : {fps_output}")
print(f"Time plot : {time_output}")