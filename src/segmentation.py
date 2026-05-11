from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# Farbmodus-Funktion
# =========================================================
def convert_color(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif mode == "r":
        return frame[:, :, 2]

    elif mode == "g":
        return frame[:, :, 1]

    elif mode == "b":
        return frame[:, :, 0]

    elif mode == "hsv_v":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]

    elif mode == "hsv_s":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]

    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# =========================================================
# ROI-HILFSFUNKTION
# Unterstützt:
#   [x, y, w, h]
# =========================================================
def extract_roi(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, int, int]:
    if roi is None:
        return frame, 0, 0

    x, y, w, h = map(int, roi)
    h_img, w_img = frame.shape[:2]

    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    return frame[y:y + h, x:x + w], x, y


# =========================================================
# INNER PIPE SEGMENTER
# =========================================================
class InnerPipeSegmenter:
    """
    Detektion des inneren Rohrs innerhalb eines ROI.

    ROI-Format:
        (x, y, w, h)

    Rückgabe:
        full_mask, detections

    detections:
        [
            {
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "area": area,
                "label": "inner_pipe",
                "contour": contour_global,
            }
        ]
    """

    def __init__(
        self,
        min_area: float = 100.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        canny_low: int = 50,
        canny_high: int = 150,
        morphology_kernel_size: int = 3,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi
        self.color_mode = color_mode

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        kernel = np.ones(
            (self.morphology_kernel_size, self.morphology_kernel_size),
            np.uint8,
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: List[Dict[str, Any]] = []
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if not contours:
            full_mask[
                y_offset:y_offset + edges.shape[0],
                x_offset:x_offset + edges.shape[1],
            ] = edges
            return full_mask, detections

        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Rohr ist typischerweise eher länglich/horizontal.
            aspect_ratio = w / max(h, 1)

            if aspect_ratio < 1.5:
                continue

            candidates.append((area, contour))

        if not candidates:
            full_mask[
                y_offset:y_offset + edges.shape[0],
                x_offset:x_offset + edges.shape[1],
            ] = edges
            return full_mask, detections

        _, best_contour = max(candidates, key=lambda item: item[0])

        x, y, w, h = cv2.boundingRect(best_contour)

        x_global = x + x_offset
        y_global = y + y_offset
        cx = x_global + w / 2.0
        cy = y_global + h / 2.0

        contour_global = best_contour.copy()
        contour_global[:, 0, 0] += x_offset
        contour_global[:, 0, 1] += y_offset

        full_mask[
            y_offset:y_offset + edges.shape[0],
            x_offset:x_offset + edges.shape[1],
        ] = edges

        detections.append(
            {
                "bbox": (int(x_global), int(y_global), int(w), int(h)),
                "center": (float(cx), float(cy)),
                "area": float(w * h),
                "label": "inner_pipe",
                "contour": contour_global,
            }
        )

        return full_mask, detections


# =========================================================
# PIPE SEGMENTER
# =========================================================
class PipeSegmenterCV:

    def __init__(
        self,
        min_area: float = 300.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        use_morphology: bool = True,
        morphology_kernel_size: int = 5,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi
        self.color_mode = color_mode

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        return extract_roi(frame, self.roi)

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

        gray = convert_color(roi_frame, self.color_mode)
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
        full_mask[
            y_offset:y_offset + mask.shape[0],
            x_offset:x_offset + mask.shape[1],
        ] = mask

        return full_mask, detections


# =========================================================
# BED SEGMENTER
# =========================================================
class BedSegmenterCV:

    def __init__(
        self,
        min_area: float = 150.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        use_morphology: bool = True,
        morphology_kernel_size: int = 3,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi
        self.color_mode = color_mode

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        return extract_roi(frame, self.roi)

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

        gray = convert_color(roi_frame, self.color_mode)
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
        full_mask[
            y_offset:y_offset + mask.shape[0],
            x_offset:x_offset + mask.shape[1],
        ] = mask

        return full_mask, detections


# =========================================================
# DETECTRON2
# =========================================================
class Detectron2Segmenter:
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

        if weights_file == "COCO":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        else:
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
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
        scores = instances.scores.numpy() if instances.has("scores") else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            w = x2 - x1
            h = y2 - y1

            if masks is not None:
                full_mask[masks[i]] = 255

            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "center": (float(x1 + w / 2.0), float(y1 + h / 2.0)),
                    "area": float(w * h),
                    "label": "detectron2_object",
                    "class_id": int(classes[i]) if classes is not None else None,
                    "score": float(scores[i]) if scores is not None else None,
                }
            )

        return full_mask, detections