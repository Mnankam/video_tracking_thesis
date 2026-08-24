#!/usr/bin/env python3
"""
evaluate_extraction_success_all_methods.py
==========================================

Repository-oriented batch evaluation of extraction success for:

- OpenCV / classical segmentation (when corresponding result files are found)
- Detectron2
- Lucas-Kanade sparse optical flow
- Farneback dense optical flow

Why this version exists
-----------------------
The repository contains several CPU/GPU benchmark and partial runs for the
same video. A naive recursive glob therefore counts the same video several
times and can incorrectly divide a short test run (e.g. 185 processed frame
transitions) by the total source-video length.

This script fixes that problem by:

1. discovering candidate result files,
2. evaluating every candidate,
3. selecting exactly ONE canonical candidate per (video_id, method),
   preferring the candidate with the largest observed coverage,
4. separating:
       - result coverage,
       - success among represented/processed units,
       - effective extraction relative to the complete source video,
5. identifying full-length versus partial runs,
6. producing an audit table so the selected source file is reproducible.

Important terminology
---------------------
expected_units
    Number of theoretically available evaluation units.
    For frame-based segmentation:
        N_expected = N_frames
    For optical flow:
        N_expected = N_frames - 1

observed_units
    Number of unique frames/frame-transitions represented by the selected
    result file.

successful_units
    Number of observed units satisfying the method-specific validity rule.

result_coverage_percent
    observed_units / expected_units * 100

conditional_success_percent
    successful_units / observed_units * 100
    This quantifies success only within units represented by the result file.

effective_extraction_rate_percent
    successful_units / expected_units * 100
    This is only interpretable as a complete-video extraction rate when the
    selected run is full-length.

A run is considered full-length when observed_units reaches a configurable
fraction of expected_units (default: 99 %).

For Lucas-Kanade, a frame transition is successful when at least
--lk-min-valid-points valid tracking points are available.

The script deliberately does NOT turn a short benchmark/test run into a low
"extraction success" value for the complete video. Partial runs are marked as
such and excluded from the thesis-oriented full-length aggregate.

Author: Serge Kouomnankam
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("evaluate_extraction_success_all_methods")

VIDEO_RE = re.compile(r"(GX\d+)", re.IGNORECASE)


# =============================================================================
# General helpers
# =============================================================================


def percentage(numerator: int | float, denominator: int | float) -> float:
    """Return percentage or NaN for a non-positive denominator."""
    if denominator is None or float(denominator) <= 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def finite_float(value: Any) -> float:
    """Convert value to float, returning NaN on failure."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def extract_video_id(path: Path) -> str | None:
    """Extract a GXxxxxxx identifier from a filename/path."""
    match = VIDEO_RE.search(path.name)
    if match is None:
        match = VIDEO_RE.search(str(path))
    if match is None:
        return None
    return match.group(1).upper()


def is_excluded_auxiliary_file(path: Path) -> bool:
    """Return True for files that are not canonical frame-level outputs."""
    name = path.name.lower()
    excluded_tokens = (
        "benchmark",
        "summary",
        "timeseries",
        "acceleration",
        "comparison",
        "metrics",
        "report",
    )
    return any(token in name for token in excluded_tokens)


def read_metadata(path: Path) -> pd.DataFrame:
    """
    Load video metadata and normalize video_id, num_frames and fps.

    The existing repository metadata uses fields compatible with this alias
    resolution, including the known video_metadata.csv.
    """
    frame = pd.read_csv(path)

    aliases = {
        "video_id": [
            "video_id",
            "video",
            "stem",
            "name",
            "filename",
            "file_name",
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

    resolved: dict[str, str] = {}

    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in frame.columns:
                resolved[target] = candidate
                break

    if "video_id" not in resolved:
        raise KeyError(
            "No video-ID column found in metadata. "
            f"Available columns: {list(frame.columns)}"
        )

    if "num_frames" not in resolved:
        raise KeyError(
            "No frame-count column found in metadata. "
            f"Available columns: {list(frame.columns)}"
        )

    video_id_series = (
        frame[resolved["video_id"]]
        .astype(str)
        .map(lambda value: Path(value).stem)
        .str.upper()
    )

    # If filename/stem contains additional text, retain the GX identifier.
    video_id_series = video_id_series.map(
        lambda value: (
            VIDEO_RE.search(value).group(1).upper()
            if VIDEO_RE.search(value)
            else value
        )
    )

    result = pd.DataFrame(
        {
            "video_id": video_id_series,
            "num_frames": pd.to_numeric(
                frame[resolved["num_frames"]],
                errors="coerce",
            ),
        }
    )

    if "fps" in resolved:
        result["fps"] = pd.to_numeric(
            frame[resolved["fps"]],
            errors="coerce",
        )
    else:
        result["fps"] = np.nan

    result = result.dropna(subset=["video_id", "num_frames"]).copy()
    result["num_frames"] = result["num_frames"].astype(int)

    # One metadata row per video.
    result = (
        result.sort_values("video_id")
        .drop_duplicates("video_id", keep="first")
        .reset_index(drop=True)
    )

    return result


def add_common_metrics(
    result: dict[str, Any],
    *,
    expected: int,
    observed: int,
    successful: int,
    full_length_threshold: float,
) -> dict[str, Any]:
    """Attach common coverage/success metrics to one candidate result."""
    observed = max(0, int(observed))
    successful = max(0, int(successful))
    expected = max(0, int(expected))

    # Guard against malformed outputs.
    successful = min(successful, observed) if observed > 0 else 0

    result["expected_units"] = expected
    result["observed_units"] = observed
    result["successful_units"] = successful

    result["result_coverage_percent"] = percentage(observed, expected)
    result["conditional_success_percent"] = percentage(successful, observed)
    result["effective_extraction_rate_percent"] = percentage(
        successful,
        expected,
    )

    result["missing_expected_units"] = max(expected - observed, 0)
    result["failed_observed_units"] = max(observed - successful, 0)

    threshold_units = int(math.ceil(expected * full_length_threshold))
    is_full = expected > 0 and observed >= threshold_units

    result["full_length_threshold_percent"] = (
        100.0 * full_length_threshold
    )
    result["is_full_length_run"] = bool(is_full)

    return result


# =============================================================================
# Lucas-Kanade
# =============================================================================


def evaluate_lucas_kanade(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
    min_valid_points: int,
    full_length_threshold: float,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    required = {"frame", "point_id", "tracking_status"}
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(
            f"Lucas-Kanade columns missing: {sorted(missing)}"
        )

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=list(required)).copy()

    if df.empty:
        raise ValueError("Lucas-Kanade result file contains no valid rows.")

    df["frame"] = df["frame"].astype(int)
    df["point_id"] = df["point_id"].astype(int)
    df["tracking_status"] = df["tracking_status"].astype(int)

    expected = max(source_frames - 1, 0)
    observed = int(df["frame"].nunique())

    per_frame = (
        df.groupby("frame")
        .agg(
            total_points=("point_id", "count"),
            valid_points=("tracking_status", "sum"),
        )
        .reset_index()
    )

    successful = int(
        (per_frame["valid_points"] >= min_valid_points).sum()
    )

    total_points = int(len(df))
    valid_points = int((df["tracking_status"] == 1).sum())

    result: dict[str, Any] = {
        "video_id": video_id,
        "method": "lucas_kanade",
        "source_frames": int(source_frames),
        "success_definition": (
            f">={min_valid_points} valid LK points per frame transition"
        ),
        "coverage_semantics": (
            "frame transitions represented in point-level LK output"
        ),
        "point_observations": total_points,
        "valid_point_observations": valid_points,
        "point_success_rate_percent": percentage(
            valid_points,
            total_points,
        ),
        "min_valid_points": int(min_valid_points),
        "median_valid_points_per_frame": float(
            per_frame["valid_points"].median()
        ),
        "min_valid_points_per_frame": int(
            per_frame["valid_points"].min()
        ),
        "max_valid_points_per_frame": int(
            per_frame["valid_points"].max()
        ),
    }

    valid = df.loc[df["tracking_status"] == 1].copy()

    if "fb_error" in valid.columns:
        fb = pd.to_numeric(valid["fb_error"], errors="coerce").dropna()
        if not fb.empty:
            result["median_fb_error_px"] = float(fb.median())
            result["p95_fb_error_px"] = float(np.percentile(fb, 95))

    if "jump_px" in valid.columns:
        jump = pd.to_numeric(valid["jump_px"], errors="coerce").dropna()
        if not jump.empty:
            result["median_jump_px"] = float(jump.median())
            result["p95_jump_px"] = float(np.percentile(jump, 95))

    if "compute_time_s" in df.columns:
        compute = pd.to_numeric(
            df["compute_time_s"],
            errors="coerce",
        ).dropna()
        if not compute.empty:
            result["median_compute_time_s"] = float(compute.median())

    return add_common_metrics(
        result,
        expected=expected,
        observed=observed,
        successful=successful,
        full_length_threshold=full_length_threshold,
    )


# =============================================================================
# Farneback
# =============================================================================


def evaluate_farneback(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
    full_length_threshold: float,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    if "frame" not in df.columns:
        raise KeyError("Farneback output has no 'frame' column.")

    frame_number = pd.to_numeric(df["frame"], errors="coerce")

    candidate_motion_columns = [
        "mean_dx",
        "mean_dy",
        "median_dx",
        "median_dy",
        "flow_magnitude",
        "mean_flow_magnitude",
        "median_flow_magnitude",
        "dx",
        "dy",
    ]

    available = [
        column
        for column in candidate_motion_columns
        if column in df.columns
    ]

    if not available:
        raise KeyError(
            "No recognized Farneback motion columns found. "
            f"Available columns: {list(df.columns)}"
        )

    numeric_motion = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce")
            for column in available
        }
    )

    represented = frame_number.notna()
    valid = represented & numeric_motion.notna().any(axis=1)

    observed = int(frame_number[represented].dropna().nunique())
    successful = int(frame_number[valid].dropna().nunique())
    expected = max(source_frames - 1, 0)

    result: dict[str, Any] = {
        "video_id": video_id,
        "method": "farneback",
        "source_frames": int(source_frames),
        "success_definition": (
            "frame transition represented with >=1 finite Farneback "
            "motion descriptor"
        ),
        "coverage_semantics": (
            "frame transitions represented in Farneback output"
        ),
        "motion_columns_used": ";".join(available),
    }

    return add_common_metrics(
        result,
        expected=expected,
        observed=observed,
        successful=successful,
        full_length_threshold=full_length_threshold,
    )


# =============================================================================
# Detectron2
# =============================================================================


def evaluate_detectron2(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
    full_length_threshold: float,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    if "frame" not in df.columns:
        raise KeyError("Detectron2 output has no 'frame' column.")

    frames = pd.to_numeric(df["frame"], errors="coerce")

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

    represented = frames.notna()
    observed = int(frames[represented].dropna().nunique())

    if success_column is not None:
        raw = df[success_column]

        if raw.dtype == bool:
            success_mask = raw.fillna(False)
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().any():
                success_mask = numeric.fillna(0) > 0
            else:
                # Conservative text parsing.
                normalized = raw.astype(str).str.strip().str.lower()
                success_mask = normalized.isin(
                    {"1", "true", "ok", "valid", "success", "detected"}
                )

        successful = int(
            frames[represented & success_mask]
            .dropna()
            .nunique()
        )
        success_definition = f"{success_column} indicates success"

    elif "score" in df.columns:
        scores = pd.to_numeric(df["score"], errors="coerce")
        successful = int(
            frames[represented & scores.notna()]
            .dropna()
            .nunique()
        )
        success_definition = (
            "frame represented by at least one row with finite Detectron2 score"
        )

    else:
        # If the canonical result schema stores only successful target
        # detections, every represented frame is a successful result frame.
        successful = observed
        success_definition = (
            "represented frame in Detectron2 result file "
            "(no explicit success/status column available)"
        )

    result: dict[str, Any] = {
        "video_id": video_id,
        "method": "detectron2",
        "source_frames": int(source_frames),
        "success_definition": success_definition,
        "coverage_semantics": (
            "frames represented in Detectron2 result file; "
            "this is not automatically identical to frames processed"
        ),
        "success_column": success_column or "",
    }

    return add_common_metrics(
        result,
        expected=int(source_frames),
        observed=observed,
        successful=successful,
        full_length_threshold=full_length_threshold,
    )


# =============================================================================
# OpenCV / classical segmentation
# =============================================================================


def evaluate_opencv(
    path: Path,
    *,
    video_id: str,
    source_frames: int,
    full_length_threshold: float,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    if "frame" not in df.columns:
        raise KeyError("OpenCV output has no 'frame' column.")

    frames = pd.to_numeric(df["frame"], errors="coerce")
    represented = frames.notna()
    observed = int(frames[represented].dropna().nunique())

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

    if success_column is not None:
        raw = df[success_column]

        if raw.dtype == bool:
            success_mask = raw.fillna(False)
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().any():
                success_mask = numeric.fillna(0) > 0
            else:
                normalized = raw.astype(str).str.strip().str.lower()
                success_mask = normalized.isin(
                    {"1", "true", "ok", "valid", "success", "detected"}
                )

        successful = int(
            frames[represented & success_mask]
            .dropna()
            .nunique()
        )
        success_definition = f"{success_column} indicates success"

    else:
        # Conservative fallback: if an OpenCV result table contains one row
        # per valid extracted frame and no explicit status, treat represented
        # frames as successful but retain the semantics in the output.
        successful = observed
        success_definition = (
            "represented frame in OpenCV result file "
            "(no explicit success/status column available)"
        )

    result: dict[str, Any] = {
        "video_id": video_id,
        "method": "opencv",
        "source_frames": int(source_frames),
        "success_definition": success_definition,
        "coverage_semantics": (
            "frames represented in OpenCV/classical segmentation output"
        ),
        "success_column": success_column or "",
    }

    return add_common_metrics(
        result,
        expected=int(source_frames),
        observed=observed,
        successful=successful,
        full_length_threshold=full_length_threshold,
    )


# =============================================================================
# Candidate discovery
# =============================================================================


def discover_candidates(outputs_root: Path) -> dict[str, list[Path]]:
    """
    Discover candidate files using strict canonical filename rules.

    Auxiliary benchmark/summary/timeseries files are excluded.
    """

    candidates: dict[str, list[Path]] = {
        "lucas_kanade": [],
        "farneback": [],
        "detectron2": [],
        "opencv": [],
    }

    for path in outputs_root.rglob("*.csv"):
        if not path.is_file():
            continue

        name = path.name.lower()

        # Avoid reading the evaluator's own results if output_dir lives under
        # outputs_root.
        if "extraction_success_all_methods" in str(path).lower():
            continue

        if is_excluded_auxiliary_file(path):
            # *_lucas_kanade_timeseries.csv and benchmark files must not be
            # interpreted as independent runs.
            continue

        if name.endswith("_lucas_kanade.csv"):
            candidates["lucas_kanade"].append(path)
            continue

        if name.endswith("_farneback_dense.csv"):
            candidates["farneback"].append(path)
            continue

        if name.endswith("_detectron2_results.csv"):
            candidates["detectron2"].append(path)
            continue

        # OpenCV outputs are less standardized in the repository. Restrict
        # discovery to paths whose directory or filename explicitly identifies
        # OpenCV/classical segmentation.
        path_lower = str(path).lower()
        if (
            "opencv" in path_lower
            or "classical" in path_lower
            or "classic_seg" in path_lower
        ):
            if extract_video_id(path) is not None:
                candidates["opencv"].append(path)

    for method in candidates:
        candidates[method] = sorted(set(candidates[method]))

    return candidates


# =============================================================================
# Candidate evaluation and canonical selection
# =============================================================================


def evaluate_candidate(
    method: str,
    path: Path,
    *,
    video_id: str,
    source_frames: int,
    lk_min_valid_points: int,
    full_length_threshold: float,
) -> dict[str, Any]:

    if method == "lucas_kanade":
        return evaluate_lucas_kanade(
            path,
            video_id=video_id,
            source_frames=source_frames,
            min_valid_points=lk_min_valid_points,
            full_length_threshold=full_length_threshold,
        )

    if method == "farneback":
        return evaluate_farneback(
            path,
            video_id=video_id,
            source_frames=source_frames,
            full_length_threshold=full_length_threshold,
        )

    if method == "detectron2":
        return evaluate_detectron2(
            path,
            video_id=video_id,
            source_frames=source_frames,
            full_length_threshold=full_length_threshold,
        )

    if method == "opencv":
        return evaluate_opencv(
            path,
            video_id=video_id,
            source_frames=source_frames,
            full_length_threshold=full_length_threshold,
        )

    raise ValueError(f"Unsupported method: {method}")


def canonical_sort_key(row: dict[str, Any]) -> tuple:
    """
    Rank candidates for the same video/method.

    Priority:
    1. full-length run,
    2. larger observed coverage,
    3. more successful units,
    4. larger point count (LK),
    5. newer modification time,
    6. stable path tie-breaker.
    """
    return (
        int(bool(row.get("is_full_length_run", False))),
        int(row.get("observed_units", 0)),
        int(row.get("successful_units", 0)),
        int(row.get("point_observations", 0) or 0),
        float(row.get("source_mtime", 0.0) or 0.0),
        str(row.get("source_path", "")),
    )


def select_canonical_candidates(
    candidate_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select exactly one candidate per (method, video_id).

    Returns
    -------
    selected_frame, audit_frame
    """
    if not candidate_rows:
        return pd.DataFrame(), pd.DataFrame()

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in candidate_rows:
        key = (str(row["method"]), str(row["video_id"]))
        groups.setdefault(key, []).append(row)

    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for (method, video_id), rows in sorted(groups.items()):
        ranked = sorted(rows, key=canonical_sort_key, reverse=True)
        selected = ranked[0]

        selected_copy = dict(selected)
        selected_copy["candidate_count_for_video_method"] = len(ranked)
        selected_rows.append(selected_copy)

        for rank, row in enumerate(ranked, start=1):
            audit = dict(row)
            audit["candidate_rank"] = rank
            audit["selected"] = rank == 1
            audit["candidate_count_for_video_method"] = len(ranked)
            audit["selection_reason"] = (
                "highest full-length status / observed coverage / "
                "successful units / point count"
                if rank == 1
                else "not selected; lower canonical ranking"
            )
            audit_rows.append(audit)

    selected_frame = pd.DataFrame(selected_rows)
    audit_frame = pd.DataFrame(audit_rows)

    return selected_frame, audit_frame


# =============================================================================
# Summaries
# =============================================================================


def summarize_methods(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build method-level summary.

    Thesis-oriented weighted extraction rate is computed ONLY from selected
    full-length runs. Partial runs are reported separately.
    """
    rows: list[dict[str, Any]] = []

    for method, subset in results.groupby("method"):
        subset = subset.copy()

        full = subset.loc[subset["is_full_length_run"] == True].copy()  # noqa: E712
        partial = subset.loc[subset["is_full_length_run"] != True].copy()  # noqa: E712

        all_expected = int(subset["expected_units"].sum())
        all_observed = int(subset["observed_units"].sum())
        all_successful = int(subset["successful_units"].sum())

        if not full.empty:
            full_expected = int(full["expected_units"].sum())
            full_observed = int(full["observed_units"].sum())
            full_successful = int(full["successful_units"].sum())

            full_effective_rate = percentage(
                full_successful,
                full_expected,
            )
            full_conditional_rate = percentage(
                full_successful,
                full_observed,
            )

            full_rates = pd.to_numeric(
                full["effective_extraction_rate_percent"],
                errors="coerce",
            ).dropna()

            mean_full = float(full_rates.mean()) if not full_rates.empty else float("nan")
            median_full = float(full_rates.median()) if not full_rates.empty else float("nan")
            min_full = float(full_rates.min()) if not full_rates.empty else float("nan")
            max_full = float(full_rates.max()) if not full_rates.empty else float("nan")
        else:
            full_expected = 0
            full_observed = 0
            full_successful = 0
            full_effective_rate = float("nan")
            full_conditional_rate = float("nan")
            mean_full = float("nan")
            median_full = float("nan")
            min_full = float("nan")
            max_full = float("nan")

        rows.append(
            {
                "method": method,
                "selected_videos": int(subset["video_id"].nunique()),
                "full_length_videos": int(full["video_id"].nunique()),
                "partial_videos": int(partial["video_id"].nunique()),
                "candidate_files_selected": int(len(subset)),
                "all_selected_expected_units": all_expected,
                "all_selected_observed_units": all_observed,
                "all_selected_successful_units": all_successful,
                "all_selected_result_coverage_percent": percentage(
                    all_observed,
                    all_expected,
                ),
                # Main thesis-oriented numbers:
                "full_length_expected_units": full_expected,
                "full_length_observed_units": full_observed,
                "full_length_successful_units": full_successful,
                "weighted_full_length_extraction_rate_percent": (
                    full_effective_rate
                ),
                "weighted_full_length_conditional_success_percent": (
                    full_conditional_rate
                ),
                "mean_full_length_video_rate_percent": mean_full,
                "median_full_length_video_rate_percent": median_full,
                "min_full_length_video_rate_percent": min_full,
                "max_full_length_video_rate_percent": max_full,
            }
        )

    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def write_bar_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Create a compact method-level full-length extraction-rate plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib unavailable; summary plot skipped.")
        return

    plot_frame = summary.copy()
    values = pd.to_numeric(
        plot_frame["weighted_full_length_extraction_rate_percent"],
        errors="coerce",
    )

    valid = values.notna()
    plot_frame = plot_frame.loc[valid].copy()
    values = values.loc[valid]

    if plot_frame.empty:
        LOGGER.warning(
            "No full-length method summary values available; plot skipped."
        )
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(plot_frame["method"], values)
    ax.set_ylabel("Full-length extraction rate [%]")
    ax.set_xlabel("Method")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for index, value in enumerate(values):
        ax.text(
            index,
            min(float(value) + 1.5, 102.0),
            f"{float(value):.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate extraction success across methods while deduplicating "
            "multiple CPU/GPU/partial runs per video."
        )
    )

    parser.add_argument(
        "--outputs-root",
        type=Path,
        required=True,
        help="Repository outputs root.",
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="video_metadata.csv containing source frame counts.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for evaluation tables and summary files.",
    )

    parser.add_argument(
        "--lk-min-valid-points",
        type=int,
        default=2,
        help=(
            "Minimum number of valid LK points required for one successful "
            "frame transition. Default: 2."
        ),
    )

    parser.add_argument(
        "--full-length-threshold",
        type=float,
        default=0.99,
        help=(
            "Minimum observed/expected fraction for a run to count as "
            "full-length. Default: 0.99."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.outputs_root.exists():
        raise FileNotFoundError(args.outputs_root)

    if not args.metadata_csv.exists():
        raise FileNotFoundError(args.metadata_csv)

    if args.lk_min_valid_points < 1:
        raise ValueError("--lk-min-valid-points must be >= 1.")

    if not (0.0 < args.full_length_threshold <= 1.0):
        raise ValueError(
            "--full-length-threshold must be in the interval (0, 1]."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(args.metadata_csv)
    metadata_by_video = metadata.set_index("video_id")

    LOGGER.info("Metadata videos: %d", len(metadata))

    discovered = discover_candidates(args.outputs_root)

    for method, paths in discovered.items():
        LOGGER.info("%s: %d candidate files", method, len(paths))

    candidate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for method, paths in discovered.items():
        for number, path in enumerate(paths, start=1):
            video_id = extract_video_id(path)

            if video_id is None:
                failures.append(
                    {
                        "video_id": "",
                        "method": method,
                        "path": str(path),
                        "error": "No GX video identifier found.",
                    }
                )
                continue

            if video_id not in metadata_by_video.index:
                failures.append(
                    {
                        "video_id": video_id,
                        "method": method,
                        "path": str(path),
                        "error": "Video not found in metadata.",
                    }
                )
                continue

            source_frames_value = metadata_by_video.loc[
                video_id,
                "num_frames",
            ]

            # Defensive handling in case metadata index is not unique.
            if isinstance(source_frames_value, pd.Series):
                source_frames_value = source_frames_value.iloc[0]

            source_frames = int(source_frames_value)

            try:
                result = evaluate_candidate(
                    method,
                    path,
                    video_id=video_id,
                    source_frames=source_frames,
                    lk_min_valid_points=args.lk_min_valid_points,
                    full_length_threshold=args.full_length_threshold,
                )

                result["source_path"] = str(path)
                result["source_parent"] = path.parent.name
                result["source_mtime"] = float(path.stat().st_mtime)
                candidate_rows.append(result)

                if number % 100 == 0:
                    LOGGER.info(
                        "%s: evaluated %d/%d candidate files",
                        method,
                        number,
                        len(paths),
                    )

            except Exception as exc:
                LOGGER.warning(
                    "Failed candidate %s / %s: %s",
                    method,
                    path,
                    exc,
                )
                failures.append(
                    {
                        "video_id": video_id,
                        "method": method,
                        "path": str(path),
                        "error": str(exc),
                    }
                )

    if not candidate_rows:
        raise RuntimeError(
            "No method/video candidate could be evaluated successfully."
        )

    selected, audit = select_canonical_candidates(candidate_rows)

    if selected.empty:
        raise RuntimeError("No canonical method/video result could be selected.")

    selected = selected.sort_values(
        ["method", "video_id"]
    ).reset_index(drop=True)

    audit = audit.sort_values(
        ["method", "video_id", "candidate_rank"]
    ).reset_index(drop=True)

    summary = summarize_methods(selected)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    selected.to_csv(
        args.output_dir / "per_video_results.csv",
        index=False,
    )

    audit.to_csv(
        args.output_dir / "candidate_audit.csv",
        index=False,
    )

    summary.to_csv(
        args.output_dir / "method_summary.csv",
        index=False,
    )

    failure_frame = pd.DataFrame(
        failures,
        columns=["video_id", "method", "path", "error"],
    )
    failure_frame.to_csv(
        args.output_dir / "failures.csv",
        index=False,
    )

    # Partial-run table is useful because these runs must not silently enter
    # the thesis aggregate.
    partial = selected.loc[
        selected["is_full_length_run"] != True  # noqa: E712
    ].copy()

    partial.to_csv(
        args.output_dir / "partial_runs.csv",
        index=False,
    )

    full_length = selected.loc[
        selected["is_full_length_run"] == True  # noqa: E712
    ].copy()

    full_length.to_csv(
        args.output_dir / "full_length_runs.csv",
        index=False,
    )

    write_bar_plot(
        summary,
        args.output_dir / "extraction_rate_by_method.png",
    )

    payload = {
        "schema_version": 2,
        "outputs_root": str(args.outputs_root),
        "metadata_csv": str(args.metadata_csv),
        "lk_min_valid_points": int(args.lk_min_valid_points),
        "full_length_threshold": float(args.full_length_threshold),
        "candidate_files_evaluated": int(len(candidate_rows)),
        "canonical_video_method_results": int(len(selected)),
        "full_length_selected_results": int(len(full_length)),
        "partial_selected_results": int(len(partial)),
        "failures": int(len(failures)),
        "method_summary": summary.to_dict(orient="records"),
    }

    with (args.output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    # -------------------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------------------

    print()
    print("=" * 120)
    print("METHOD SUMMARY -- CANONICAL RUN PER VIDEO/METHOD")
    print("=" * 120)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("RUN SELECTION")
    print("=" * 120)
    print(f"Candidate files evaluated : {len(candidate_rows)}")
    print(f"Canonical results selected: {len(selected)}")
    print(f"Full-length selected runs : {len(full_length)}")
    print(f"Partial selected runs     : {len(partial)}")
    print(f"Failures                  : {len(failures)}")

    print()
    print("Important:")
    print(
        "  Thesis-wide extraction rates should be taken from "
        "'weighted_full_length_extraction_rate_percent'."
    )
    print(
        "  Partial runs are listed separately in partial_runs.csv and are "
        "not used for that full-length aggregate."
    )

    print()
    print("Output files:")
    for filename in (
        "per_video_results.csv",
        "full_length_runs.csv",
        "partial_runs.csv",
        "candidate_audit.csv",
        "method_summary.csv",
        "failures.csv",
        "summary.json",
        "extraction_rate_by_method.png",
    ):
        print(f"  {args.output_dir / filename}")

    # Sanity-check GX010129 if available.
    sanity = selected.loc[
        (selected["video_id"] == "GX010129")
        & (selected["method"] == "lucas_kanade")
    ]

    if not sanity.empty:
        row = sanity.iloc[0]
        print()
        print("=" * 120)
        print("GX010129 LUCAS-KANADE SANITY CHECK")
        print("=" * 120)
        print(f"Selected path                 : {row['source_path']}")
        print(f"Expected transitions          : {int(row['expected_units'])}")
        print(f"Observed transitions          : {int(row['observed_units'])}")
        print(f"Successful transitions        : {int(row['successful_units'])}")
        print(
            "Effective extraction rate [%]: "
            f"{float(row['effective_extraction_rate_percent']):.6f}"
        )
        if pd.notna(row.get("point_success_rate_percent", np.nan)):
            print(
                "Point success rate [%]       : "
                f"{float(row['point_success_rate_percent']):.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())