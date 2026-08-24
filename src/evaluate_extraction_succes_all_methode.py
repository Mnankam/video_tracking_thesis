#!/usr/bin/env python3
"""
evaluate_extraction_success_all_methods.py

Batch evaluation of frame-level extraction success for:

- OpenCV segmentation
- Detectron2 segmentation
- Lucas-Kanade sparse optical flow
- Farneback dense optical flow

The script evaluates every available recording and produces both
per-video and aggregated method-level statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(
    "evaluate_extraction_success_all_methods"
)


# =============================================================================
# General helpers
# =============================================================================


def read_metadata(path: Path) -> pd.DataFrame:
    """
    Load video metadata.

    Required logical information:
        video_id
        frame count
        frame rate
    """

    frame = pd.read_csv(path)

    # Adapt aliases to the actual metadata file.
    aliases = {
        "video_id": [
            "video_id",
            "video",
            "stem",
            "name",
        ],
        "num_frames": [
            "num_frames",
            "nb_frames",
            "frames",
            "frame_count",
        ],
        "fps": [
            "fps",
            "frame_rate",
            "sampling_rate",
        ],
    }

    resolved = {}

    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in frame.columns:
                resolved[target] = candidate
                break

    if "video_id" not in resolved:
        raise KeyError(
            "No video-ID column found in metadata."
        )

    if "num_frames" not in resolved:
        raise KeyError(
            "No frame-count column found in metadata."
        )

    result = pd.DataFrame(
        {
            "video_id": (
                frame[
                    resolved["video_id"]
                ]
                .astype(str)
                .str.replace(
                    ".MP4",
                    "",
                    regex=False,
                )
            ),
            "num_frames": pd.to_numeric(
                frame[
                    resolved["num_frames"]
                ],
                errors="coerce",
            ),
        }
    )

    if "fps" in resolved:
        result["fps"] = pd.to_numeric(
            frame[
                resolved["fps"]
            ],
            errors="coerce",
        )
    else:
        result["fps"] = np.nan

    result = result.dropna(
        subset=[
            "video_id",
            "num_frames",
        ]
    )

    result["num_frames"] = (
        result["num_frames"]
        .astype(int)
    )

    return result


def percentage(
    successful: int,
    expected: int,
) -> float:
    if expected <= 0:
        return float("nan")

    return (
        100.0
        * successful
        / expected
    )


# =============================================================================
# Lucas-Kanade
# =============================================================================


def evaluate_lucas_kanade(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
    min_valid_points: int,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    required = {
        "frame",
        "point_id",
        "tracking_status",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise KeyError(
            f"Lucas-Kanade columns missing: "
            f"{sorted(missing)}"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=list(required)
    )

    df["frame"] = (
        df["frame"]
        .astype(int)
    )

    df["tracking_status"] = (
        df["tracking_status"]
        .astype(int)
    )

    expected = (
        source_frames - 1
    )

    observed_frames = int(
        df["frame"].nunique()
    )

    per_frame = (
        df.groupby("frame")
        .agg(
            total_points=(
                "point_id",
                "count",
            ),
            valid_points=(
                "tracking_status",
                "sum",
            ),
        )
    )

    successful = int(
        (
            per_frame[
                "valid_points"
            ]
            >= min_valid_points
        ).sum()
    )

    total_points = int(
        len(df)
    )

    valid_points = int(
        (
            df[
                "tracking_status"
            ]
            == 1
        ).sum()
    )

    result = {
        "video_id": video_id,
        "method": "lucas_kanade",
        "source_frames": source_frames,
        "expected_units": expected,
        "observed_units": observed_frames,
        "successful_units": successful,
        "failed_units": (
            expected - successful
        ),
        "extraction_rate_percent": percentage(
            successful,
            expected,
        ),
        "point_observations": total_points,
        "valid_point_observations": valid_points,
        "point_success_rate_percent": percentage(
            valid_points,
            total_points,
        ),
        "min_valid_points": (
            min_valid_points
        ),
    }

    if "fb_error" in df.columns:
        fb = pd.to_numeric(
            df.loc[
                df["tracking_status"] == 1,
                "fb_error",
            ],
            errors="coerce",
        ).dropna()

        if not fb.empty:
            result[
                "median_fb_error_px"
            ] = float(
                fb.median()
            )

            result[
                "p95_fb_error_px"
            ] = float(
                np.percentile(
                    fb,
                    95,
                )
            )

    if "jump_px" in df.columns:
        jump = pd.to_numeric(
            df.loc[
                df["tracking_status"] == 1,
                "jump_px",
            ],
            errors="coerce",
        ).dropna()

        if not jump.empty:
            result[
                "median_jump_px"
            ] = float(
                jump.median()
            )

            result[
                "p95_jump_px"
            ] = float(
                np.percentile(
                    jump,
                    95,
                )
            )

    return result


# =============================================================================
# Farneback
# =============================================================================


def evaluate_farneback(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    if "frame" not in df.columns:
        raise KeyError(
            "Farneback output has no 'frame' column."
        )

    frame_number = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    # Possible motion-output columns.
    candidate_motion_columns = [
        "mean_dx",
        "mean_dy",
        "median_dx",
        "median_dy",
        "flow_magnitude",
        "mean_flow_magnitude",
    ]

    available = [
        column
        for column in candidate_motion_columns
        if column in df.columns
    ]

    if not available:
        raise KeyError(
            "No recognized Farneback motion columns found."
        )

    valid = frame_number.notna()

    # A processed transition is considered usable when at least
    # one required motion descriptor is finite.
    numeric_motion = pd.DataFrame(
        {
            column: pd.to_numeric(
                df[column],
                errors="coerce",
            )
            for column in available
        }
    )

    valid &= (
        numeric_motion
        .notna()
        .any(axis=1)
    )

    successful_frames = int(
        frame_number[
            valid
        ].nunique()
    )

    observed_frames = int(
        frame_number
        .dropna()
        .nunique()
    )

    expected = (
        source_frames - 1
    )

    return {
        "video_id": video_id,
        "method": "farneback",
        "source_frames": source_frames,
        "expected_units": expected,
        "observed_units": observed_frames,
        "successful_units": successful_frames,
        "failed_units": (
            expected
            - successful_frames
        ),
        "extraction_rate_percent": percentage(
            successful_frames,
            expected,
        ),
    }


# =============================================================================
# OpenCV segmentation
# =============================================================================


def evaluate_opencv(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    if "frame" not in df.columns:
        raise KeyError(
            "OpenCV output has no 'frame' column."
        )

    frames = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    # IMPORTANT:
    # Adapt these candidates to the actual OpenCV result schema.
    validity_candidates = [
        "inner_pipe_detected",
        "inner_pipe_valid",
        "segmentation_valid",
        "detection_success",
        "status",
    ]

    success_column = next(
        (
            column
            for column in validity_candidates
            if column in df.columns
        ),
        None,
    )

    if success_column is None:
        raise KeyError(
            "No explicit OpenCV success column found. "
            "Do not infer success only from row existence; "
            "adapt evaluate_opencv() to the real output schema."
        )

    success = df[
        success_column
    ]

    if success.dtype == bool:
        success = success.astype(int)

    success = pd.to_numeric(
        success,
        errors="coerce",
    ).fillna(0)

    successful_frames = int(
        frames[
            success > 0
        ]
        .dropna()
        .nunique()
    )

    observed_frames = int(
        frames
        .dropna()
        .nunique()
    )

    expected = source_frames

    return {
        "video_id": video_id,
        "method": "opencv",
        "source_frames": source_frames,
        "expected_units": expected,
        "observed_units": observed_frames,
        "successful_units": successful_frames,
        "failed_units": (
            expected
            - successful_frames
        ),
        "extraction_rate_percent": percentage(
            successful_frames,
            expected,
        ),
    }


# =============================================================================
# Detectron2
# =============================================================================


def evaluate_detectron2(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    if "frame" not in df.columns:
        raise KeyError(
            "Detectron2 output has no 'frame' column."
        )

    frames = pd.to_numeric(
        df["frame"],
        errors="coerce",
    )

    # Prefer an explicit detection-success flag.
    success_candidates = [
        "detection_success",
        "segmentation_valid",
        "detected",
        "status",
    ]

    success_column = next(
        (
            column
            for column in success_candidates
            if column in df.columns
        ),
        None,
    )

    if success_column is not None:
        success = pd.to_numeric(
            df[success_column],
            errors="coerce",
        ).fillna(0)

        successful_frames = int(
            frames[
                success > 0
            ]
            .dropna()
            .nunique()
        )

    elif "score" in df.columns:
        # This branch should only be used if each row represents
        # a valid target-class detection.
        scores = pd.to_numeric(
            df["score"],
            errors="coerce",
        )

        successful_frames = int(
            frames[
                scores.notna()
            ]
            .dropna()
            .nunique()
        )

    else:
        raise KeyError(
            "No explicit Detectron2 success indicator found. "
            "Adapt evaluate_detectron2() to the repository output schema."
        )

    observed_frames = int(
        frames
        .dropna()
        .nunique()
    )

    expected = source_frames

    return {
        "video_id": video_id,
        "method": "detectron2",
        "source_frames": source_frames,
        "expected_units": expected,
        "observed_units": observed_frames,
        "successful_units": successful_frames,
        "failed_units": (
            expected
            - successful_frames
        ),
        "extraction_rate_percent": percentage(
            successful_frames,
            expected,
        ),
    }


# =============================================================================
# Method summary
# =============================================================================


def summarize_methods(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for method, subset in results.groupby(
        "method"
    ):

        total_expected = int(
            subset[
                "expected_units"
            ].sum()
        )

        total_successful = int(
            subset[
                "successful_units"
            ].sum()
        )

        rates = pd.to_numeric(
            subset[
                "extraction_rate_percent"
            ],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "method": method,
                "num_videos": int(
                    len(subset)
                ),
                "total_expected_units": (
                    total_expected
                ),
                "total_successful_units": (
                    total_successful
                ),
                "weighted_extraction_rate_percent": (
                    percentage(
                        total_successful,
                        total_expected,
                    )
                ),
                "mean_video_rate_percent": float(
                    rates.mean()
                ),
                "median_video_rate_percent": float(
                    rates.median()
                ),
                "min_video_rate_percent": float(
                    rates.min()
                ),
                "max_video_rate_percent": float(
                    rates.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frame-level extraction success "
            "for OpenCV, Detectron2, Lucas-Kanade "
            "and Farneback over all available recordings."
        )
    )

    parser.add_argument(
        "--outputs-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--lk-min-valid-points",
        type=int,
        default=2,
    )

    return parser


def main() -> int:

    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = read_metadata(
        args.metadata_csv
    )

    results = []
    failures = []

    # -------------------------------------------------------------------------
    # File discovery
    # -------------------------------------------------------------------------
    #
    # These patterns must match the actual repository output structure.
    # Lucas-Kanade is already known from the current data.
    #

    method_patterns = {
        "lucas_kanade": (
            "**/*_lucas_kanade.csv"
        ),
        "farneback": (
            "**/*farneback*.csv"
        ),
        "opencv": (
            "**/*opencv*.csv"
        ),
        "detectron2": (
            "**/*detectron2*.csv"
        ),
    }

    metadata_by_video = (
        metadata
        .set_index(
            "video_id"
        )
    )

    for method, pattern in method_patterns.items():

        paths = sorted(
            args.outputs_root.glob(
                pattern
            )
        )

        LOGGER.info(
            "%s: found %d candidate files",
            method,
            len(paths),
        )

        for path in paths:

            # Extract GXxxxxxx identifier from filename.
            video_id = next(
                (
                    part
                    for part in path.stem.split("_")
                    if part.startswith("GX")
                ),
                None,
            )

            if video_id is None:
                continue

            if (
                video_id
                not in metadata_by_video.index
            ):
                failures.append(
                    {
                        "video_id": video_id,
                        "method": method,
                        "path": str(path),
                        "error": (
                            "Video not found in metadata."
                        ),
                    }
                )
                continue

            source_frames = int(
                metadata_by_video.loc[
                    video_id,
                    "num_frames",
                ]
            )

            try:

                if method == "lucas_kanade":
                    result = evaluate_lucas_kanade(
                        path,
                        video_id=video_id,
                        source_frames=source_frames,
                        min_valid_points=(
                            args.lk_min_valid_points
                        ),
                    )

                elif method == "farneback":
                    result = evaluate_farneback(
                        path,
                        video_id=video_id,
                        source_frames=source_frames,
                    )

                elif method == "opencv":
                    result = evaluate_opencv(
                        path,
                        video_id=video_id,
                        source_frames=source_frames,
                    )

                elif method == "detectron2":
                    result = evaluate_detectron2(
                        path,
                        video_id=video_id,
                        source_frames=source_frames,
                    )

                else:
                    continue

                result[
                    "source_path"
                ] = str(path)

                results.append(
                    result
                )

            except Exception as exc:

                LOGGER.exception(
                    "Failed: %s / %s",
                    video_id,
                    method,
                )

                failures.append(
                    {
                        "video_id": video_id,
                        "method": method,
                        "path": str(path),
                        "error": str(exc),
                    }
                )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    result_frame = pd.DataFrame(
        results
    )

    if result_frame.empty:
        raise RuntimeError(
            "No method/video combination was evaluated successfully."
        )

    result_frame = result_frame.sort_values(
        [
            "method",
            "video_id",
        ]
    )

    result_frame.to_csv(
        args.output_dir
        / "per_video_results.csv",
        index=False,
    )

    summary = summarize_methods(
        result_frame
    )

    summary.to_csv(
        args.output_dir
        / "method_summary.csv",
        index=False,
    )

    pd.DataFrame(
        failures
    ).to_csv(
        args.output_dir
        / "failures.csv",
        index=False,
    )

    with (
        args.output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            {
                "number_of_successful_evaluations": int(
                    len(result_frame)
                ),
                "number_of_failures": int(
                    len(failures)
                ),
                "lk_min_valid_points": int(
                    args.lk_min_valid_points
                ),
                "method_summary": (
                    summary
                    .to_dict(
                        orient="records"
                    )
                ),
            },
            handle,
            indent=2,
        )

    print()
    print("=" * 100)
    print("METHOD SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        f"Output: {args.output_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )