from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class BedEdgeDetector:
    """
    Detektion der Bettkante innerhalb eines ROI.

    Strategie:
    - ROI ausschneiden
    - Graubild / HSV analysieren
    - Bettbereich segmentieren
    - obere Bettkante bestimmen
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        blur_kernel: Tuple[int, int] = (9, 9),
        threshold_value: int = 90,
        morphology_kernel_size: int = 5,
        color_mode: str = "gray",
    ) -> None:

        self.roi = roi
        self.blur_kernel = blur_kernel
        self.threshold_value = threshold_value
        self.morphology_kernel_size = morphology_kernel_size
        self.color_mode = color_mode

    # =====================================================
    # Farbmodus
    # =====================================================
    def _convert_color(self, frame: np.ndarray) -> np.ndarray:

        if self.color_mode == "gray":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elif self.color_mode == "g":
            return frame[:, :, 1]

        elif self.color_mode == "hsv_v":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 2]

        elif self.color_mode == "hsv_s":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 1]

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # Hauptfunktion
    # =====================================================
    def detect(self, frame: np.ndarray) -> Dict[str, float]:

        h, w = frame.shape[:2]

        # =================================================
        # ROI extrahieren
        # =================================================
        if self.roi is not None:
            x, y, rw, rh = self.roi

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))

            roi_frame = frame[y:y + rh, x:x + rw]

            x_offset = x
            y_offset = y

        else:
            roi_frame = frame
            x_offset = 0
            y_offset = 0

        # =================================================
        # Vorverarbeitung
        # =================================================
        gray = self._convert_color(roi_frame)

        gray = cv2.GaussianBlur(
            gray,
            self.blur_kernel,
            0,
        )

        # =================================================
        # Threshold
        # =================================================
        _, mask = cv2.threshold(
            gray,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY,
        )

        # =================================================
        # Morphologie
        # =================================================
        kernel = np.ones(
            (
                self.morphology_kernel_size,
                self.morphology_kernel_size,
            ),
            np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # =================================================
        # Vertikales Profil
        # =================================================
        row_profile = np.mean(mask, axis=1)

        # erste Zeile mit starkem Signal suchen
        threshold_profile = 20

        indices = np.where(row_profile > threshold_profile)[0]

        if len(indices) == 0:

            return {
                "success": False,
                "y_edge": np.nan,
                "x_left": float(x_offset),
                "x_right": float(x_offset + roi_frame.shape[1] - 1),
            }

        # =================================================
        # obere Bettkante
        # =================================================
        y_local = int(indices[0])

        # globale Koordinate
        y_edge_global = y_offset + y_local

        return {
            "success": True,
            "y_edge": float(y_edge_global),
            "x_left": float(x_offset),
            "x_right": float(x_offset + roi_frame.shape[1] - 1),
        }
