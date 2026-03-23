from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class SimpleSegmenter:
    def __init__(
        self,
        min_area: float = 50.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        use_morphology: bool = True,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.use_morphology = use_morphology

    def _preprocess_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        return gray

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.use_morphology:
            return mask

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _extract_detections(self, mask: np.ndarray) -> List[Dict[str, Any]]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Dict[str, Any]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2.0
            cy = y + h / 2.0

            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "center": (float(cx), float(cy)),
                    "area": area,
                    "contour": contour,
                }
            )

        return detections

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        gray = self._preprocess_gray(frame)

        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        mask = self._postprocess_mask(mask)
        detections = self._extract_detections(mask)

        return mask, detections


class Detectron2Segmenter:
    """
    Detectron2-basierte Instanzsegmentierung.
    Erwartet funktionierende Detectron2-Installation im HPC/Container.
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
                }
            )

        return full_mask, detections