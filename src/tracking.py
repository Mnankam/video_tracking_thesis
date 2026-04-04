from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from filterpy.kalman import KalmanFilter


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# =========================================================
# 1. SINGLE OBJECT TRACKER (für Innenrohr)
# =========================================================
class SingleObjectTracker:
    """
    Speziell für Innenrohr:
    - nur ein Objekt
    - stabiler Kalman-Filter
    """

    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.initialized = False

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

        self.kf.P *= 10
        self.kf.R *= 5
        self.kf.Q *= 0.01

    def update(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not detections:
            return None

        # größtes Objekt = Rohr
        best = max(detections, key=lambda d: d["area"])
        cx, cy = best["center"]

        if not self.initialized:
            self.kf.x = np.array([cx, cy, 0, 0], dtype=float)
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(np.array([cx, cy]))

        px, py = float(self.kf.x[0]), float(self.kf.x[1])

        return {
            "track_id": 1,
            "bbox": best["bbox"],
            "center": (px, py),
            "area": best["area"],
            "label": "inner_pipe",
        }


# =========================================================
# 2. MULTI OBJECT TRACKER (für Partikel)
# =========================================================
class KalmanTrack:
    def __init__(self, track_id: int, detection: Dict[str, Any]) -> None:
        self.track_id = track_id
        self.bbox = detection["bbox"]
        self.center = detection["center"]
        self.area = detection["area"]
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

        self.kf.P *= 10
        self.kf.R *= 5
        self.kf.Q *= 0.01

    def predict(self):
        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])

    def update(self, detection):
        cx, cy = detection["center"]
        self.kf.update(np.array([cx, cy]))

        self.bbox = detection["bbox"]
        self.center = (float(self.kf.x[0]), float(self.kf.x[1]))
        self.area = detection["area"]
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1

    def current(self):
        return float(self.kf.x[0]), float(self.kf.x[1])


class MultiObjectTracker:
    def __init__(self, max_distance=60, max_missed=5):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: List[KalmanTrack] = []
        self.next_id = 1

    def update(self, detections):
        predictions = [t.predict() for t in self.tracks]

        matches = []
        used = set()

        for i, pred in enumerate(predictions):
            best = None
            best_dist = 1e9

            for j, det in enumerate(detections):
                if j in used:
                    continue
                dist = euclidean_distance(pred, det["center"])
                if dist < best_dist and dist < self.max_distance:
                    best = j
                    best_dist = dist

            if best is not None:
                matches.append((i, best))
                used.add(best)

        matched_tracks = set()
        matched_dets = set()

        for t, d in matches:
            self.tracks[t].update(detections[d])
            matched_tracks.add(t)
            matched_dets.add(d)

        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.mark_missed()

        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(KalmanTrack(self.next_id, det))
                self.next_id += 1

        results = []
        for t in self.tracks:
            cx, cy = t.current()
            results.append({
                "track_id": t.track_id,
                "bbox": t.bbox,
                "center": (cx, cy),
                "area": t.area,
                "label": "particle",
            })

        return results