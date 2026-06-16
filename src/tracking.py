from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# =========================================================
# 1. SINGLE OBJECT TRACKER
# =========================================================
class SingleObjectTracker:
    """
    Stabiler Kalman-Tracker für ein dominantes Objekt:
    - inner_pipe
    - pipe
    - particle_bed
    """

    def __init__(self, max_missed: int = 10):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.initialized = False
        self.missed_frames = 0
        self.max_missed = max_missed

        self.last_bbox = None
        self.last_area = None
        self.last_label = "object"

        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.kf.P *= 10.0
        self.kf.R *= 5.0
        self.kf.Q *= 0.01

    def update(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if detections:
            best = max(detections, key=lambda d: d.get("area", 0.0))
            cx, cy = best["center"]

            if not self.initialized:
                self.kf.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
                self.initialized = True
            else:
                self.kf.predict()
                self.kf.update(np.array([cx, cy]))

            self.last_bbox = best["bbox"]
            self.last_area = best.get("area", 0.0)
            self.last_label = best.get("label", "object")
            self.missed_frames = 0

        else:
            if not self.initialized:
                return None

            self.kf.predict()
            self.missed_frames += 1

            if self.missed_frames > self.max_missed:
                return None

        px, py = float(self.kf.x[0]), float(self.kf.x[1])

        return {
            "track_id": 1,
            "bbox": self.last_bbox,
            "center": (px, py),
            "area": self.last_area,
            "label": self.last_label,
            "missed_frames": self.missed_frames,
        }


# =========================================================
# 2. MULTI OBJECT TRACKER
# =========================================================
class KalmanTrack:
    def __init__(self, track_id: int, detection: Dict[str, Any]) -> None:
        self.track_id = track_id
        self.bbox = detection["bbox"]
        self.center = detection["center"]
        self.area = detection.get("area", 0.0)
        self.label = detection.get("label", "object")
        self.missed_frames = 0

        cx, cy = self.center

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([cx, cy, 0.0, 0.0])

        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.kf.P *= 10.0
        self.kf.R *= 5.0
        self.kf.Q *= 0.01

    def predict(self):
        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])

    def update(self, detection):
        cx, cy = detection["center"]
        self.kf.update(np.array([cx, cy]))

        self.bbox = detection["bbox"]
        self.center = (float(self.kf.x[0]), float(self.kf.x[1]))
        self.area = detection.get("area", self.area)
        self.label = detection.get("label", self.label)
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1

    def current(self):
        return float(self.kf.x[0]), float(self.kf.x[1])


class MultiObjectTracker:
    def __init__(self, max_distance=60.0, max_missed=5):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: List[KalmanTrack] = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]):
        predictions = [t.predict() for t in self.tracks]

        matches = []
        used_detections = set()

        for i, pred in enumerate(predictions):
            best_det = None
            best_dist = float("inf")

            for j, det in enumerate(detections):
                if j in used_detections:
                    continue

                dist = euclidean_distance(pred, det["center"])

                if dist < best_dist and dist < self.max_distance:
                    best_det = j
                    best_dist = dist

            if best_det is not None:
                matches.append((i, best_det))
                used_detections.add(best_det)

        matched_tracks = set()
        matched_dets = set()

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()

        self.tracks = [
            t for t in self.tracks
            if t.missed_frames <= self.max_missed
        ]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(KalmanTrack(self.next_id, det))
                self.next_id += 1

        results = []

        for track in self.tracks:
            cx, cy = track.current()

            results.append({
                "track_id": track.track_id,
                "bbox": track.bbox,
                "center": (cx, cy),
                "area": track.area,
                "label": track.label,
                "missed_frames": track.missed_frames,
            })

        return results