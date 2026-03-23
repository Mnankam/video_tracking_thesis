from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


class KalmanTrack:
    def __init__(self, track_id: int, detection: Dict[str, Any]) -> None:
        self.track_id = track_id
        self.bbox = detection["bbox"]
        self.center = detection["center"]
        self.area = detection["area"]
        self.missed_frames = 0

        cx, cy = self.center

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([cx, cy, 0.0, 0.0], dtype=float)

        self.kf.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )

        self.kf.P *= 10.0
        self.kf.R *= 5.0
        self.kf.Q *= 0.01

    def predict(self) -> Tuple[float, float]:
        self.kf.predict()
        pred_x, pred_y = self.kf.x[0], self.kf.x[1]
        return float(pred_x), float(pred_y)

    def update(self, detection: Dict[str, Any]) -> None:
        cx, cy = detection["center"]
        self.kf.update(np.array([cx, cy], dtype=float))

        self.bbox = detection["bbox"]
        self.center = (float(self.kf.x[0]), float(self.kf.x[1]))
        self.area = detection["area"]
        self.missed_frames = 0

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def current_prediction(self) -> Tuple[float, float]:
        return float(self.kf.x[0]), float(self.kf.x[1])


class KalmanTracker:
    """
    Einfacher Multi-Object Tracker mit:
    - Kalman-Filter pro Track
    - Greedy Matching über Distanz
    - Track-Löschung nach max_missed Frames
    """

    def __init__(
        self,
        max_distance: float = 60.0,
        max_missed: int = 5,
    ) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: List[KalmanTrack] = []
        self.next_id = 1

    def _predict_all(self) -> List[Tuple[float, float]]:
        predictions = []
        for track in self.tracks:
            predictions.append(track.predict())
        return predictions

    def _match_detections(
        self,
        detections: List[Dict[str, Any]],
        predictions: List[Tuple[float, float]],
    ) -> List[Tuple[int, int]]:
        matches = []
        used_detections = set()

        for t_idx, pred_center in enumerate(predictions):
            best_dist = float("inf")
            best_det_idx = -1

            for d_idx, det in enumerate(detections):
                if d_idx in used_detections:
                    continue

                dist = euclidean_distance(pred_center, det["center"])
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_det_idx = d_idx

            if best_det_idx >= 0:
                matches.append((t_idx, best_det_idx))
                used_detections.add(best_det_idx)

        return matches

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = self._predict_all()
        matches = self._match_detections(detections, predictions)

        matched_tracks = set()
        matched_detections = set()

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])
            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.mark_missed()

        self.tracks = [
            track for track in self.tracks if track.missed_frames <= self.max_missed
        ]

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                self.tracks.append(KalmanTrack(self.next_id, det))
                self.next_id += 1

        results = []
        for track in self.tracks:
            x, y, w, h = track.bbox
            cx, cy = track.current_prediction()
            results.append(
                {
                    "track_id": track.track_id,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "area": track.area,
                }
            )

        return results