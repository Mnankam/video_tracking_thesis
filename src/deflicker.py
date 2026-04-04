from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class BasicDeflicker:
    """
    Einfache globale Helligkeitsnormalisierung als Baseline.

    Die Idee hierfür ist:
    - Referenzhelligkeit aus dem ersten Frame zu setzen
    - alle weiteren Frames darauf skalieren
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


@dataclass
class FFTDeflicker:
    """
    Frequenzbasierter Deflicker für Hochgeschwindigkeitsvideos.

    die Idee hierfür ist:
    - Helligkeitsverlauf über die Zeit aufzubauen
    - dominante Flickerkomponente bei bekannter Frequenz (z. B. 50 Hz) zu schätzen
    - aktuelle Helligkeit gezielt kompensieren
    """

    fps: float = 200.0
    flicker_freq: float = 50.0
    use_second_harmonic: bool = False
    window_size: int = 256
    min_history: int = 32
    smooth_alpha: float = 0.2
    use_median: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None

    brightness_history: List[float] = field(default_factory=list)
    reference_brightness: Optional[float] = None
    smoothed_target: Optional[float] = None

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _measure_brightness(self, frame: np.ndarray) -> float:
        roi_frame = self._extract_roi(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        if self.use_median:
            return float(np.median(gray))
        return float(np.mean(gray))

    def _update_history(self, brightness: float) -> None:
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > self.window_size:
            self.brightness_history.pop(0)

    def _estimate_reference(self) -> float:
        if len(self.brightness_history) < 8:
            return self.brightness_history[-1]
        return float(np.mean(self.brightness_history[: min(20, len(self.brightness_history))]))

    def _compute_flicker_component(self) -> Optional[np.ndarray]:
        if len(self.brightness_history) < self.min_history:
            return None

        x = np.asarray(self.brightness_history, dtype=np.float32)
        x_centered = x - np.mean(x)

        spectrum = np.fft.rfft(x_centered)
        freqs = np.fft.rfftfreq(len(x_centered), d=1.0 / self.fps)

        filtered = np.zeros_like(spectrum)

        # Hauptfrequenz
        idx = int(np.argmin(np.abs(freqs - self.flicker_freq)))
        if idx > 0:
            filtered[idx] = spectrum[idx]

            if idx - 1 >= 0:
                filtered[idx - 1] = 0.5 * spectrum[idx - 1]
            if idx + 1 < len(filtered):
                filtered[idx + 1] = 0.5 * spectrum[idx + 1]

        # Optional zweite Harmonische
        if self.use_second_harmonic:
            harmonic_freq = 2.0 * self.flicker_freq
            idx2 = int(np.argmin(np.abs(freqs - harmonic_freq)))
            if 0 < idx2 < len(filtered):
                filtered[idx2] = 0.5 * spectrum[idx2]
                if idx2 - 1 >= 0:
                    filtered[idx2 - 1] = 0.25 * spectrum[idx2 - 1]
                if idx2 + 1 < len(filtered):
                    filtered[idx2 + 1] = 0.25 * spectrum[idx2 + 1]

        if not np.any(np.abs(filtered) > 0):
            return None

        flicker_component = np.fft.irfft(filtered, n=len(x_centered)).real
        return flicker_component

    def apply(self, frame: np.ndarray) -> np.ndarray:
        brightness = self._measure_brightness(frame)
        self._update_history(brightness)

        if self.reference_brightness is None:
            self.reference_brightness = self._estimate_reference()
            self.smoothed_target = self.reference_brightness
            return frame

        if len(self.brightness_history) < self.min_history:
            target = self.reference_brightness
        else:
            flicker_component = self._compute_flicker_component()

            if flicker_component is None:
                target = self.reference_brightness
            else:
                current_flicker_estimate = float(flicker_component[-1])
                target = self.reference_brightness - current_flicker_estimate

            if self.smoothed_target is None:
                self.smoothed_target = target
            else:
                self.smoothed_target = (
                    self.smooth_alpha * target
                    + (1.0 - self.smooth_alpha) * self.smoothed_target
                )

            target = self.smoothed_target

        if brightness < 1e-6:
            return frame

        scale = target / brightness
        corrected = np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return corrected