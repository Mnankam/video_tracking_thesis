from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import math
from collections import defaultdict


class Evaluator:
    def __init__(self) -> None:
        self.times: List[float] = []
        self.track_counts: List[int] = []
        self._start_time: float | None = None

    def start_timer(self) -> None:
        self._start_time = time.time()

    def stop_timer(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer wurde nicht gestartet. Erst start_timer() aufrufen.")
        elapsed = time.time() - self._start_time
        self.times.append(elapsed)
        self._start_time = None
        return elapsed

    def add_track_count(self, n_tracks: int) -> None:
        self.track_counts.append(n_tracks)

    def average_time(self) -> float:
        return float(np.mean(self.times)) if self.times else 0.0

    def fps(self) -> float:
        avg = self.average_time()
        return float(1.0 / avg) if avg > 0 else 0.0

    def average_track_count(self) -> float:
        return float(np.mean(self.track_counts)) if self.track_counts else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "avg_processing_time_s": self.average_time(),
            "avg_pipeline_fps": self.fps(),
            "avg_track_count": self.average_track_count(),
            "num_processed_frames": float(len(self.times)),
        }


# ================================
# EXISTIERENDE METRIKEN
# ================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0

    return float(np.sum(intersection) / union_sum)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.sum(np.logical_and(mask1_bool, mask2_bool))
    total = np.sum(mask1_bool) + np.sum(mask2_bool)

    if total == 0:
        return 0.0

    return float(2.0 * intersection / total)


# ================================
# NEUE METRIKEN
# ================================

def compute_runtime_metrics(frame_times):
    frame_times = np.array(frame_times, dtype=float)

    mean_time = float(np.mean(frame_times))
    std_time = float(np.std(frame_times))
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    return {
        "mean_frame_time_s": mean_time,
        "std_frame_time_s": std_time,
        "fps": fps
    }


def compute_tracking_metrics(frame_tracks, jump_threshold_px=50.0):
    track_lengths = defaultdict(int)
    track_positions = defaultdict(list)

    for frame_idx, tracks in enumerate(frame_tracks):
        for tr in tracks:
            tid = tr["id"]
            cx = float(tr["cx"])
            cy = float(tr["cy"])

            track_lengths[tid] += 1
            track_positions[tid].append((frame_idx, cx, cy))

    if not track_lengths:
        return {
            "average_track_length": 0.0,
            "num_unique_tracks": 0,
            "num_large_jumps": 0,
            "large_jump_ratio": 0.0
        }

    avg_track_length = float(np.mean(list(track_lengths.values())))
    num_unique_tracks = len(track_lengths)

    num_large_jumps = 0
    num_total_transitions = 0

    for tid, positions in track_positions.items():
        positions = sorted(positions, key=lambda x: x[0])

        for i in range(1, len(positions)):
            _, x1, y1 = positions[i - 1]
            _, x2, y2 = positions[i]
            dist = math.hypot(x2 - x1, y2 - y1)

            num_total_transitions += 1
            if dist > jump_threshold_px:
                num_large_jumps += 1

    large_jump_ratio = (
        num_large_jumps / num_total_transitions
        if num_total_transitions > 0 else 0.0
    )

    return {
        "average_track_length": avg_track_length,
        "num_unique_tracks": num_unique_tracks,
        "num_large_jumps": num_large_jumps,
        "large_jump_ratio": large_jump_ratio
    }


def compute_temporal_iou_metrics(masks):
    if len(masks) < 2:
        return {
            "mean_temporal_iou": 0.0,
            "std_temporal_iou": 0.0
        }

    ious = []

    for i in range(len(masks) - 1):
        a = masks[i].astype(bool)
        b = masks[i + 1].astype(bool)

        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou)

    return {
        "mean_temporal_iou": float(np.mean(ious)),
        "std_temporal_iou": float(np.std(ious))
    }


def compute_bed_edge_metrics(bed_edge_values, jump_threshold_px=20):
    vals = np.array(bed_edge_values, dtype=float)

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))

    diffs = np.abs(np.diff(vals))
    num_large_jumps = int(np.sum(diffs > jump_threshold_px))

    return {
        "mean_bed_edge_y": mean_val,
        "std_bed_edge_y": std_val,
        "num_large_bed_jumps": num_large_jumps
    }


# ================================
# HAUPT FUNKTION
# ================================

def evaluate_all_metrics(
    frame_times,
    frame_tracks,
    masks=None,
    bed_edge_values=None
):
    results = {}

    results["runtime"] = compute_runtime_metrics(frame_times)
    results["tracking"] = compute_tracking_metrics(frame_tracks)

    if masks is not None and len(masks) > 1:
        results["temporal_iou"] = compute_temporal_iou_metrics(masks)

    if bed_edge_values is not None and len(bed_edge_values) > 0:
        results["bed_edge"] = compute_bed_edge_metrics(bed_edge_values)

    return results


# ================================
# TRACKING STATS 
# ================================

@dataclass
class TrackingStats:
    frame_to_num_tracks: List[int] = field(default_factory=list)
    seen_track_ids: Set[int] = field(default_factory=set)
    total_track_observations: int = 0

    def update(self, tracks: List[Dict[str, Any]]) -> None:
        self.frame_to_num_tracks.append(len(tracks))
        self.total_track_observations += len(tracks)

        for track in tracks:
            self.seen_track_ids.add(int(track["track_id"]))

    def to_dict(self) -> Dict[str, float]:
        avg_tracks = (
            float(np.mean(self.frame_to_num_tracks))
            if self.frame_to_num_tracks
            else 0.0
        )

        max_tracks = (
            float(np.max(self.frame_to_num_tracks))
            if self.frame_to_num_tracks
            else 0.0
        )

        return {
            "avg_tracks_per_frame": avg_tracks,
            "max_tracks_in_frame": max_tracks,
            "num_unique_track_ids": float(len(self.seen_track_ids)),
            "total_track_observations": float(self.total_track_observations),
        }


def merge_summaries(*summaries: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for summary in summaries:
        merged.update(summary)
    return merged


def save_summary_csv(summary: Dict[str, Any], output_csv: str) -> None:
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])