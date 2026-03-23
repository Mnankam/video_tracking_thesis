from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np


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


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Berechnet Intersection over Union (IoU) zweier Binärmasken.
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0

    return float(np.sum(intersection) / union_sum)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Berechnet den Dice Score zweier Binärmasken.
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.sum(np.logical_and(mask1_bool, mask2_bool))
    total = np.sum(mask1_bool) + np.sum(mask2_bool)

    if total == 0:
        return 0.0

    return float(2.0 * intersection / total)


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
    """
    Führt mehrere Summary-Dictionaries zu einem zusammen.
    Spätere Einträge überschreiben frühere Schlüssel.
    """
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