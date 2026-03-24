from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class BedEdgeDetector:
    """
    Einfache Detektion der Bettkante über das vertikale Intensitätsprofil.
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        blur_kernel: Tuple[int, int] = (9, 9),
    ) -> None:
        self.roi = roi
        self.blur_kernel = blur_kernel

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        h, w = frame.shape[:2]

        if self.roi is not None:
            x, y, rw, rh = self.roi
            roi_frame = frame[y:y + rh, x:x + rw]
            x_offset = x
            y_offset = y
        else:
            roi_frame = frame
            x_offset = 0
            y_offset = 0

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        row_profile = np.mean(gray, axis=1)
        gradient = np.gradient(row_profile)

        y_local = int(np.argmax(np.abs(gradient)))
        y_edge = y_local + y_offset

        return {
            "y_edge": float(y_edge),
            "x_left": float(x_offset),
            "x_right": float(x_offset + gray.shape[1] - 1),
        }