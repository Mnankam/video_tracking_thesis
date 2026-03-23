from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


class Deflicker:
    """
    Einfache Helligkeitsnormalisierung zur Reduktion von Flicker (z.B. 50 Hz).

    Idee:
    - Referenzhelligkeit wird aus erstem Frame gesetzt
    - Alle weiteren Frames werden daran angepasst
    """

    def __init__(self) -> None:
        self.reference_brightness: Optional[float] = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = float(np.mean(gray))

        if self.reference_brightness is None:
            self.reference_brightness = current_brightness
            return frame

        if current_brightness < 1e-6:
            return frame

        scale = self.reference_brightness / current_brightness
        corrected = np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        return corrected