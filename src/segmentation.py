from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# COLOR CONVERSION
# =========================================================
def convert_color(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "r":
        return frame[:, :, 2]

    if mode == "g":
        return frame[:, :, 1]

    if mode == "b":
        return frame[:, :, 0]

    if mode == "hsv_h":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 0]

    if mode == "hsv_s":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]

    if mode == "hsv_v":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# =========================================================
# ROI HELPER
# ROI format: [x, y, w, h]
# =========================================================
def extract_roi(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, int, int]:
    if roi is None:
        return frame, 0, 0

    x, y, w, h = map(int, roi)
    img_h, img_w = frame.shape[:2]

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    return frame[y:y + h, x:x + w], x, y


def make_full_mask(
    frame: np.ndarray,
    roi_mask: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[
        y_offset:y_offset + roi_mask.shape[0],
        x_offset:x_offset + roi_mask.shape[1],
    ] = roi_mask
    return full_mask


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    open_iter: int = 1,
    close_iter: int = 2,
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return mask


def contour_to_detection(
    contour: np.ndarray,
    x_offset: int,
    y_offset: int,
    label: str,
) -> Dict[str, Any]:
    x, y, w, h = cv2.boundingRect(contour)

    x_global = x + x_offset
    y_global = y + y_offset

    cx = x_global + w / 2.0
    cy = y_global + h / 2.0

    contour_global = contour.copy()
    contour_global[:, 0, 0] += x_offset
    contour_global[:, 0, 1] += y_offset

    return {
        "bbox": (int(x_global), int(y_global), int(w), int(h)),
        "center": (float(cx), float(cy)),
        "area": float(cv2.contourArea(contour)),
        "label": label,
        "contour": contour_global,
    }


def filter_contours(
    contours: List[np.ndarray],
    min_area: float,
    min_aspect_ratio: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
) -> List[np.ndarray]:
    valid = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
            continue

        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue

        valid.append(contour)

    return valid


# =========================================================
# INNER PIPE SEGMENTER
# =========================================================
class InnerPipeSegmenter:
    """
    Detects the inner transparent pipe inside inner_pipe_roi.
    The method focuses on horizontal elongated edge structures.
    """

    def __init__(
        self,
        min_area: float = 80.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        canny_low: int = 30,
        canny_high: int = 120,
        morphology_kernel_size: int = 5,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
        min_aspect_ratio: float = 4.0,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi
        self.color_mode = color_mode
        self.min_aspect_ratio = min_aspect_ratio

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morphology_kernel_size * 3, self.morphology_kernel_size),
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = filter_contours(
            contours,
            min_area=self.min_area,
            min_aspect_ratio=self.min_aspect_ratio,
        )

        detections: List[Dict[str, Any]] = []

        if contours:
            best = max(contours, key=cv2.contourArea)
            detections.append(
                contour_to_detection(best, x_offset, y_offset, "inner_pipe")
            )

        full_mask = make_full_mask(frame, edges, x_offset, y_offset)
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

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        if self.use_morphology:
            mask = clean_mask(mask, self.morphology_kernel_size)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = filter_contours(
            contours,
            min_area=self.min_area,
            min_aspect_ratio=2.0,
        )

        detections = [
            contour_to_detection(c, x_offset, y_offset, "pipe")
            for c in contours
        ]

        full_mask = make_full_mask(frame, mask, x_offset, y_offset)
        return full_mask, detections


# =========================================================
# BED SEGMENTER
# =========================================================
class BedSegmenterCV:
    """
    Detects the particle/fluidized bed region.
    Uses HSV information when possible, because the bed is usually
    better separated in saturation/value than in pure grayscale.
    """

    def __init__(
        self,
        min_area: float = 150.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        use_morphology: bool = True,
        morphology_kernel_size: int = 5,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "hsv_s",
        threshold_mode: str = "otsu",
        invert: bool = False,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi
        self.color_mode = color_mode
        self.threshold_mode = threshold_mode
        self.invert = invert

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        if self.threshold_mode == "adaptive":
            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                -3,
            )
        else:
            thresh_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
            _, mask = cv2.threshold(
                gray,
                0,
                255,
                thresh_type + cv2.THRESH_OTSU,
            )

        return mask

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        mask = self._threshold(gray)

        if self.use_morphology:
            mask = clean_mask(
                mask,
                kernel_size=self.morphology_kernel_size,
                open_iter=1,
                close_iter=2,
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = filter_contours(
            contours,
            min_area=self.min_area,
            min_aspect_ratio=1.2,
        )

        detections = [
            contour_to_detection(c, x_offset, y_offset, "particle_bed")
            for c in contours
        ]

        full_mask = make_full_mask(frame, mask, x_offset, y_offset)
        return full_mask, detections


# =========================================================
# OPTICAL BOX SEGMENTER
# currently not required by pipeline, but usable later
# =========================================================
class OpticalBoxSegmenter:
    def __init__(
        self,
        min_area: float = 300.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
        canny_low: int = 30,
        canny_high: int = 120,
        morphology_kernel_size: int = 5,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.roi = roi
        self.color_mode = color_mode
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morphology_kernel_size = morphology_kernel_size

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morphology_kernel_size, self.morphology_kernel_size),
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = filter_contours(contours, min_area=self.min_area)

        detections = [
            contour_to_detection(c, x_offset, y_offset, "optical_box")
            for c in contours
        ]

        full_mask = make_full_mask(frame, edges, x_offset, y_offset)
        return full_mask, detections


# =========================================================
# DETECTRON2
# =========================================================
class Detectron2Segmenter:
    """
    Detectron2-based instance segmentation.
    Useful only if Detectron2 is installed inside the container.
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

        if weights_file == "COCO":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        else:
            cfg.MODEL.WEIGHTS = weights_file

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.DEVICE = device

        self.cfg = cfg
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
        scores = (
            instances.scores.numpy().tolist()
            if instances.has("scores")
            else None
        )

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            w = int(x2 - x1)
            h = int(y2 - y1)
            cx = float(x1 + w / 2.0)
            cy = float(y1 + h / 2.0)
            area = float(w * h)

            if masks is not None:
                full_mask[masks[i]] = 255

            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "center": (cx, cy),
                    "area": area,
                    "label": "detectron2_object",
                    "class_id": int(classes[i]) if classes is not None else None,
                    "score": float(scores[i]) if scores is not None else None,
                }
            )

        return full_mask, detections