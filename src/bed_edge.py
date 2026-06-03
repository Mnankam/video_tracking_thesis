from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class BedEdgeDetector:
    """
    Detektion der Bettkante innerhalb eines ROI.

    Strategie:
    - ROI ausschneiden
    - Farbe/Grauwert analysieren
    - horizontale Kanten über vertikalen Gradienten suchen
    - den unteren Teil der ROI stärker gewichten
    - Ergebnis in globale Bildkoordinaten zurückrechnen
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        blur_kernel: Tuple[int, int] = (9, 9),
        threshold_value: int = 90,
        morphology_kernel_size: int = 5,
        color_mode: str = "gray",
        prefer_lower_half: bool = True,
        lower_weight_strength: float = 1.8,
        min_signal: float = 5.0,
    ) -> None:
        self.roi = roi
        self.blur_kernel = blur_kernel
        self.threshold_value = threshold_value
        self.morphology_kernel_size = morphology_kernel_size
        self.color_mode = color_mode
        self.prefer_lower_half = prefer_lower_half
        self.lower_weight_strength = lower_weight_strength
        self.min_signal = min_signal

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

    def _extract_roi(self, frame: np.ndarray):
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

        # Vertikaler Gradient: horizontale Kanten werden betont
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_profile = np.mean(np.abs(sobel_y), axis=1)

        # Optional zusätzlich eine Maskeninformation verwenden
        _, mask = cv2.threshold(
            gray,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morphology_kernel_size * 3, self.morphology_kernel_size),
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask_profile = np.mean(mask, axis=1) / 255.0

        # Kombination aus Kantenstärke und Maskensignal
        score = gradient_profile * (1.0 + mask_profile)

        # Unteren Teil der ROI bevorzugen, damit obere Reflexionen weniger stark dominieren
        if self.prefer_lower_half:
            n = len(score)
            weights = np.linspace(1.0, self.lower_weight_strength, n)
            score = score * weights

        if len(score) == 0 or float(np.max(score)) < self.min_signal:
            return {
                "success": False,
                "y_edge": np.nan,
                "x_left": float(x_offset),
                "x_right": float(x_offset + roi_frame.shape[1] - 1),
            }

        y_local = int(np.argmax(score))
        y_edge_global = y_offset + y_local

        return {
            "success": True,
            "y_edge": float(y_edge_global),
            "x_left": float(x_offset),
            "x_right": float(x_offset + roi_frame.shape[1] - 1),
            "y_local": float(y_local),
            "score": float(np.max(score)),
        }