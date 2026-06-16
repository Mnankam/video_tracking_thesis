from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class BedEdgeDetector:
    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        blur_kernel: Tuple[int, int] = (9, 9),
        morphology_kernel_size: int = 5,
        color_mode: str = "hsv_v",
        prefer_lower_half: bool = True,
        lower_weight_strength: float = 1.8,
        min_signal: float = 12.0,
        expected_y_ratio: float = 0.65,
        expected_y_weight: float = 0.8,
    ) -> None:
        self.roi = roi
        self.blur_kernel = blur_kernel
        self.morphology_kernel_size = morphology_kernel_size
        self.color_mode = color_mode
        self.prefer_lower_half = prefer_lower_half
        self.lower_weight_strength = lower_weight_strength
        self.min_signal = min_signal
        self.expected_y_ratio = expected_y_ratio
        self.expected_y_weight = expected_y_weight

    def _convert_color(self, frame: np.ndarray) -> np.ndarray:
        if self.color_mode == "gray":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.color_mode == "g":
            return frame[:, :, 1]

        if self.color_mode == "hsv_v":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 2]

        if self.color_mode == "hsv_s":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 1]

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, w = frame.shape[:2]

        if self.roi is None:
            return frame, 0, 0

        x, y, rw, rh = self.roi

        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        rw = max(1, min(int(rw), w - x))
        rh = max(1, min(int(rh), h - y))

        roi_frame = frame[y:y + rh, x:x + rw]
        return roi_frame, x, y

    def detect(self, frame: np.ndarray) -> Dict[str, float]:
        roi_frame, x_offset, y_offset = self._extract_roi(frame)

        gray = self._convert_color(roi_frame)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_profile = np.mean(np.abs(sobel_y), axis=1)

        _, mask_binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        _, mask_binary_inv = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        mask = cv2.bitwise_or(mask_binary, mask_binary_inv)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (35, 2),
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_profile = np.mean(mask, axis=1) / 255.0

        score = gradient_profile * (1.0 + mask_profile)

        n = len(score)

        if n == 0:
            return {
                "success": False,
                "y_edge": np.nan,
                "x_left": float(x_offset),
                "x_right": float(x_offset + roi_frame.shape[1] - 1),
            }

        if self.prefer_lower_half:
            lower_weights = np.linspace(1.0, self.lower_weight_strength, n)
            score = score * lower_weights

        expected_y = self.expected_y_ratio * (n - 1)
        y_positions = np.arange(n)

        expected_weight = np.exp(
            -0.5 * ((y_positions - expected_y) / max(1.0, 0.25 * n)) ** 2
        )

        score = score * (1.0 + self.expected_y_weight * expected_weight)

        score_smooth = cv2.GaussianBlur(
            score.astype(np.float32).reshape(-1, 1),
            (1, 9),
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