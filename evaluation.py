import time
import numpy as np


class Evaluator:
    def __init__(self):
        self.times = []

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        elapsed = time.time() - self._start_time
        self.times.append(elapsed)
        return elapsed

    def average_time(self):
        return np.mean(self.times) if self.times else 0

    def fps(self):
        if not self.times:
            return 0
        return 1.0 / np.mean(self.times)


def compute_iou(mask1, mask2):
    """
    Berechnet Intersection over Union (IoU)
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    if np.sum(union) == 0:
        return 0

    return np.sum(intersection) / np.sum(union)


def evaluate_tracking(trajectories):
    """
    Einfache Tracking-Auswertung (Platzhalter)
    """
    num_tracks = len(trajectories)
    return {
        "num_tracks": num_tracks
    }