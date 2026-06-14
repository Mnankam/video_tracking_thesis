from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
    if mode == "hsv_s":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    if mode == "hsv_v":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def extract_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]):
    if roi is None:
        return frame, 0, 0

    x, y, w, h = map(int, roi)
    img_h, img_w = frame.shape[:2]

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    return frame[y:y + h, x:x + w], x, y


def make_full_mask(frame, roi_mask, x_offset, y_offset):
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[
        y_offset:y_offset + roi_mask.shape[0],
        x_offset:x_offset + roi_mask.shape[1],
    ] = roi_mask
    return full_mask


def clean_mask(mask, kernel_size=3, open_iter=1, close_iter=1, horizontal=False):
    if horizontal:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, kernel_size * 2), max(1, kernel_size // 2)),
        )
    else:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return mask


def contour_to_detection(contour, x_offset, y_offset, label):
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
    contours,
    min_area,
    min_aspect_ratio=None,
    max_aspect_ratio=None,
    min_width=None,
    min_height=None,
    max_height=None,
):
    valid = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        if min_width is not None and w < min_width:
            continue
        if min_height is not None and h < min_height:
            continue
        if max_height is not None and h > max_height:
            continue
        if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
            continue
        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue

        valid.append(contour)

    return valid

class InnerPipeSegmenter:
    def __init__(
        self,
        min_area: float = 120.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        canny_low: int = 12,
        canny_high: int = 60,
        morphology_kernel_size: int = 3,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
        min_aspect_ratio: float = 3.0,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morphology_kernel_size = morphology_kernel_size
        self.roi = roi
        self.color_mode = color_mode
        self.min_aspect_ratio = min_aspect_ratio

    def segment(self, frame):
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        _, mask = cv2.threshold(
            gray_eq,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            center_y = y + h / 2.0

            if (
                aspect_ratio > 3.0
                and 15 < center_y < 50
                and w > 120
                and 5 <= h <= 70
            ):
                candidates.append(cnt)

        detections = []

        if candidates:
            best = max(candidates, key=cv2.contourArea)

            detections.append(
                contour_to_detection(
                    best,
                    x_offset,
                    y_offset,
                    "inner_pipe",
                )
            )

            clean = np.zeros_like(mask)
            cv2.drawContours(clean, [best], -1, 255, thickness=-1)
            mask = clean
        else:
            mask = np.zeros_like(mask)

        full_mask = make_full_mask(frame, mask, x_offset, y_offset)
        return full_mask, detections

class PipeSegmenterCV:
    def __init__(
        self,
        min_area: float = 200.0,
        blur_kernel: Tuple[int, int] = (3, 3),
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

    def segment(self, frame):
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            2,
        )

        if self.use_morphology:
            mask = clean_mask(
                mask,
                kernel_size=self.morphology_kernel_size,
                open_iter=1,
                close_iter=1,
                horizontal=True,
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = filter_contours(
            contours,
            min_area=self.min_area,
            min_aspect_ratio=1.5,
            min_width=30,
        )

        detections = [
            contour_to_detection(c, x_offset, y_offset, "pipe")
            for c in contours
        ]

        full_mask = make_full_mask(frame, mask, x_offset, y_offset)
        return full_mask, detections


class BedSegmenterCV:
    def __init__(
        self,
        min_area: float = 80.0,
        blur_kernel: Tuple[int, int] = (3, 3),
        use_morphology: bool = True,
        morphology_kernel_size: int = 5,
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "hsv_v",
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

    def segment(self, frame):
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        if self.blur_kernel is not None:
            v_channel = cv2.GaussianBlur(v_channel, self.blur_kernel, 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_channel = clahe.apply(v_channel)

        _, mask = cv2.threshold(
            v_channel,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)

            # y_global ist wichtig, weil y innerhalb der ROI liegt
            y_global = y + y_offset

            if (
                y_global > 390
                and aspect_ratio >= 2.0
                and w >= 80
                and 5 <= h <= 90
            ):
                candidates.append(cnt)

        detections = []

        if candidates:
            best = max(candidates, key=cv2.contourArea)

            detections.append(
                contour_to_detection(
                    best,
                    x_offset,
                    y_offset,
                    "particle_bed",
                )
            )

            clean = np.zeros_like(mask)
            cv2.drawContours(clean, [best], -1, 255, thickness=-1)
            mask = clean
        else:
            mask = np.zeros_like(mask)

        full_mask = make_full_mask(frame, mask, x_offset, y_offset)
        return full_mask, detections
    
class OpticalBoxSegmenter:
    def __init__(
        self,
        min_area: float = 80.0,
        blur_kernel: Tuple[int, int] = (3, 3),
        roi: Optional[Tuple[int, int, int, int]] = None,
        color_mode: str = "gray",
        canny_low: int = 30,
        canny_high: int = 120,
        morphology_kernel_size: int = 3,
    ) -> None:
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.roi = roi
        self.color_mode = color_mode
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morphology_kernel_size = morphology_kernel_size

    def segment(self, frame):
        roi_frame, x_offset, y_offset = extract_roi(frame, self.roi)

        gray = convert_color(roi_frame, self.color_mode)

        if self.blur_kernel is not None:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = filter_contours(
            contours,
            min_area=self.min_area,
            min_aspect_ratio=1.2,
            min_width=10,
            min_height=2,
        )

        detections = [
            contour_to_detection(c, x_offset, y_offset, "optical_box")
            for c in contours
        ]

        if detections:
            detections = sorted(detections, key=lambda d: d["area"], reverse=True)[:3]

        full_mask = make_full_mask(frame, edges, x_offset, y_offset)
        return full_mask, detections


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

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def segment(self, frame):
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")

        height, width = frame.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        detections = []

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