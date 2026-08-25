#!/usr/bin/env python3
"""
Analyse Detectron2 results at object and frame level.

The script distinguishes between:

- detections: object-level rows;
- processed frames: frame-level rows;
- frames containing at least one Detectron2 detection.

This distinction is necessary because one frame can contain several
detections and a processed frame can contain no detection at all.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description="Analyse Detectron2 results."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Object-level Detectron2 CSV.",
    )

    parser.add_argument(
        "--frame-csv",
        required=True,
        help="Frame-level Detectron2 CSV.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory.",
    )

    args = parser.parse_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    detections = pd.read_csv(
        args.csv
    )

    frames = pd.read_csv(
        args.frame_csv
    )

    required_frame_columns = {
        "frame",
        "processed",
        "num_detections",
        "has_detection",
        "compute_time_s",
    }

    missing = (
        required_frame_columns
        - set(frames.columns)
    )

    if missing:
        raise KeyError(
            "Frame-level CSV is missing columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # Frame statistics
    # -------------------------------------------------------------------------

    num_processed_frames = int(
        len(frames)
    )

    num_frames_with_detections = int(
        frames["has_detection"].sum()
    )

    num_frames_without_detections = int(
        num_processed_frames
        - num_frames_with_detections
    )

    if num_processed_frames > 0:

        detection_output_rate = (
            100.0
            * num_frames_with_detections
            / num_processed_frames
        )

        mean_compute_time = float(
            frames["compute_time_s"].mean()
        )

        median_compute_time = float(
            frames["compute_time_s"].median()
        )

        p95_compute_time = float(
            np.percentile(
                frames["compute_time_s"],
                95,
            )
        )

        inference_fps = (
            1.0 / mean_compute_time
            if mean_compute_time > 0
            else float("nan")
        )

    else:

        detection_output_rate = float("nan")
        mean_compute_time = float("nan")
        median_compute_time = float("nan")
        p95_compute_time = float("nan")
        inference_fps = float("nan")

    # -------------------------------------------------------------------------
    # Detection statistics
    # -------------------------------------------------------------------------

    num_detections = int(
        len(detections)
    )

    if (
        not detections.empty
        and "score" in detections.columns
    ):
        mean_score = float(
            detections["score"].mean()
        )
        median_score = float(
            detections["score"].median()
        )
    else:
        mean_score = float("nan")
        median_score = float("nan")

    if (
        not detections.empty
        and "area" in detections.columns
    ):
        mean_area = float(
            detections["area"].mean()
        )
    else:
        mean_area = float("nan")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "num_processed_frames": (
            num_processed_frames
        ),
        "num_frames_with_detections": (
            num_frames_with_detections
        ),
        "num_frames_without_detections": (
            num_frames_without_detections
        ),
        "detection_output_rate_percent": (
            detection_output_rate
        ),
        "num_detections": (
            num_detections
        ),
        "mean_score": (
            mean_score
        ),
        "median_score": (
            median_score
        ),
        "mean_area_px2": (
            mean_area
        ),
        "mean_compute_time_s": (
            mean_compute_time
        ),
        "median_compute_time_s": (
            median_compute_time
        ),
        "p95_compute_time_s": (
            p95_compute_time
        ),
        "inference_fps_from_mean_time": (
            inference_fps
        ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = os.path.join(
        args.out_dir,
        "detectron2_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 80)
    print("DETECTRON2 ANALYSIS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()
    print(f"Summary saved: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()