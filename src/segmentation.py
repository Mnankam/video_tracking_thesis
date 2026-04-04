from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PipeSegmenterCV:
    """
    Klassische Segmentierung zur Bestimmung der Innenrohrposition.

    Ziel:
    - dominante Rohrstruktur im Bild lokalisieren
    - Bounding Box und Zentrum ableiten

    """

    def __init__(
        self,
        min_area: float = 300.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        use_morphology: bool = True,
        morphology_kernel_size: int = 5,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        if self.roi is None:
            return frame, 0, 0
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w], x, y

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.use_morphology:
            return mask

        kernel = np.ones(
            (self.morphology_kernel_size, self.morphology_kernel_size),
            np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = self._extract_roi(frame)

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        mask = self._postprocess_mask(mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x += x_offset
            y += y_offset

            cx = x + w / 2.0
            cy = y + h / 2.0

            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "center": (float(cx), float(cy)),
                    "area": area,
                    "label": "inner_pipe",
                }
            )

        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y_offset:y_offset + mask.shape[0], x_offset:x_offset + mask.shape[1]] = mask

        return full_mask, detections


class BedSegmenterCV:
    """
    Klassische Segmentierung für das Partikelbett.

    Ziel:
    - Bettbereich isolieren
    - optionale Ableitung von Bettgrenzen oder Bettregionen
    """

    def __init__(
        self,
        min_area: float = 150.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        use_morphology: bool = True,
        morphology_kernel_size: int = 3,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        if self.roi is None:
            return frame, 0, 0
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w], x, y

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.use_morphology:
            return mask

        kernel = np.ones(
            (self.morphology_kernel_size, self.morphology_kernel_size),
            np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = self._extract_roi(frame)

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        mask = self._postprocess_mask(mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x += x_offset
            y += y_offset

            cx = x + w / 2.0
            cy = y + h / 2.0

            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "center": (float(cx), float(cy)),
                    "area": area,
                    "label": "particle_bed",
                }
            )

        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y_offset:y_offset + mask.shape[0], x_offset:x_offset + mask.shape[1]] = mask

        return full_mask, detections


class Detectron2Segmenter:
    """
    Detectron2-basierte Instanzsegmentierung.
    Erwartet funktionierende Detectron2-Installation im HPC/Container hier wird Podman verwendet.
    """

    def __init__(
        self,
        config_file: str,
        weights_file: str,
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = weights_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(cfg)

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")

        height, width = frame.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        detections: List[Dict[str, Any]] = []

        if not instances.has("pred_boxes"):
            return full_mask, detections

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        classes = (
            instances.pred_classes.numpy().tolist()
            if instances.has("pred_classes")
            else None
        )

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            area = float(w * h)

            if masks is not None:
                full_mask[masks[i]] = 255

            detections.append(
                {
                    "bbox": (x1, y1, w, h),
                    "center": (cx, cy),
                    "area": area,
                    "class_id": classes[i] if classes is not None else None,
                }
            )

        return full_mask, detections