from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class BedEdgeDetector:
    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        blur_kernel: Tuple[int, int] = (7, 7),
        color_mode: str = "hsv_v",
        min_signal: float = 8.0,
        expected_y_ratio: float = 0.30,
        expected_y_weight: float = 1.2,
    ) -> None:
        self.roi = roi
        self.blur_kernel = blur_kernel
        self.color_mode = color_mode
        self.min_signal = min_signal
        self.expected_y_ratio = expected_y_ratio
        self.expected_y_weight = expected_y_weight

    def _convert_color(self, frame):
        if self.color_mode == "gray":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.color_mode == "g":
            return frame[:, :, 1]
        if self.color_mode == "hsv_s":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]

    def _extract_roi(self, frame):
        h, w = frame.shape[:2]

        if self.roi is None:
            return frame, 0, 0

        x, y, rw, rh = map(int, self.roi)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

        return frame[y:y + rh, x:x + rw], x, y

    def detect(self, frame) -> Dict[str, float]:
        roi_frame, x_offset, y_offset = self._extract_roi(frame)

        gray = self._convert_color(roi_frame)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        score = np.mean(np.abs(sobel_y), axis=1)

        n = len(score)

        if n == 0:
            return {
                "success": False,
                "y_edge": np.nan,
                "x_left": float(x_offset),
                "x_right": float(x_offset + roi_frame.shape[1] - 1),
            }

        expected_y = self.expected_y_ratio * (n - 1)
        y_positions = np.arange(n)

        expected_weight = np.exp(
            -0.5 * ((y_positions - expected_y) / max(1.0, 0.25 * n)) ** 2
        )

        score = score * (1.0 + self.expected_y_weight * expected_weight)

        score_smooth = cv2.GaussianBlur(
            score.astype(np.float32).reshape(-1, 1),
            (1, 7),
            0,
        ).ravel()

        if float(np.max(score_smooth)) < self.min_signal:
            return {
                "success": False,
                "y_edge": np.nan,
                "x_left": float(x_offset),
                "x_right": float(x_offset + roi_frame.shape[1] - 1),
            }

        y_local = int(np.argmax(score_smooth))
        y_edge_global = y_offset + y_local

        return {
            "success": True,
            "y_edge": float(y_edge_global),
            "x_left": float(x_offset),
            "x_right": float(x_offset + roi_frame.shape[1] - 1),
            "y_local": float(y_local),
            "score": float(np.max(score_smooth)),
        }