from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from src.deflicker import FFTDeflicker
from src.segmentation import (
    PipeSegmenterCV,
    BedSegmenterCV,
    Detectron2Segmenter,
    InnerPipeSegmenter,
    OpticalBoxSegmenter,
)
from src.bed_edge import BedEdgeDetector
from src.tracking import SingleObjectTracker, MultiObjectTracker
from src.evaluation import (
    Evaluator,
    TrackingStats,
    merge_summaries,
    save_summary_csv,
)


@dataclass
class PipelineConfig:
    video_path: str
    output_csv: str

    start_frame: int = 0
    end_frame: Optional[int] = None

    resize_width: Optional[int] = None
    resize_height: Optional[int] = None

    enable_deflicker: bool = True
    enable_tracking: bool = True
    enable_bed_edge: bool = True
    enable_inner_pipe: bool = False
    enable_optical_box: bool = True

    save_debug_frames: bool = False
    debug_dir: str = "outputs/debug"

    segmentation_mode: str = "pipe_cv"
    segmentation_color_mode: str = "gray"

    detectron2_config_file: Optional[str] = None
    detectron2_weights_file: Optional[str] = None
    detectron2_score_threshold: float = 0.5
    device: str = "cpu"

    save_summary: bool = True
    summary_csv: str = "outputs/summary.csv"

    pipe_roi: Optional[list[int]] = None
    bed_roi: Optional[list[int]] = None
    bed_edge_roi: Optional[list[int]] = None
    inner_pipe_roi: Optional[list[int]] = None
    optical_box_roi: Optional[list[int]] = None

    bed_edge_color_mode: str = "gray"
    bed_edge_smoothing: bool = True
    bed_edge_median_window: int = 5
    bed_edge_ema_alpha: float = 0.3

    deflicker_fps: float = 200.0
    deflicker_freq: float = 50.0
    deflicker_use_second_harmonic: bool = False
    deflicker_window_size: int = 256
    deflicker_min_history: int = 32
    deflicker_smooth_alpha: float = 0.4
    deflicker_use_median: bool = True
    deflicker_roi: Optional[list[int]] = None


class VideoPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.evaluator = Evaluator()
        self.tracking_stats = TrackingStats()

        self.bed_edge_history = deque(maxlen=config.bed_edge_median_window)
        self.bed_edge_ema: Optional[float] = None

        self.deflicker = None
        if config.enable_deflicker:
            self.deflicker = FFTDeflicker(
                fps=config.deflicker_fps,
                flicker_freq=config.deflicker_freq,
                use_second_harmonic=config.deflicker_use_second_harmonic,
                window_size=config.deflicker_window_size,
                min_history=config.deflicker_min_history,
                smooth_alpha=config.deflicker_smooth_alpha,
                use_median=config.deflicker_use_median,
                roi=tuple(config.deflicker_roi) if config.deflicker_roi else None,
            )

        if config.segmentation_mode == "detectron2":
            self.segmenter = Detectron2Segmenter(
                config_file=config.detectron2_config_file,
                weights_file=config.detectron2_weights_file,
                score_threshold=config.detectron2_score_threshold,
                device=config.device,
            )

        elif config.segmentation_mode == "pipe_cv":
            self.segmenter = PipeSegmenterCV(
                roi=tuple(config.pipe_roi) if config.pipe_roi else None,
                color_mode=config.segmentation_color_mode,
            )

        elif config.segmentation_mode == "bed_cv":
            self.segmenter = BedSegmenterCV(
                roi=tuple(config.bed_roi) if config.bed_roi else None,
                color_mode=config.segmentation_color_mode,
            )

        else:
            raise ValueError(f"Unbekannter segmentation_mode: {config.segmentation_mode}")

        self.inner_pipe_segmenter = None
        if config.enable_inner_pipe:
            self.inner_pipe_segmenter = InnerPipeSegmenter(
                roi=tuple(config.inner_pipe_roi) if config.inner_pipe_roi else None,
                color_mode="gray",
            )

        self.optical_box_segmenter = None
        if config.enable_optical_box:
            self.optical_box_segmenter = OpticalBoxSegmenter(
                roi=tuple(config.optical_box_roi) if config.optical_box_roi else None,
                color_mode="gray",
            )

        if config.enable_tracking:
            if config.segmentation_mode == "pipe_cv":
                self.tracker = SingleObjectTracker()
            else:
                self.tracker = MultiObjectTracker(max_distance=60.0, max_missed=5)
        else:
            self.tracker = None

        self.bed_edge_detector = None
        if config.enable_bed_edge:
            self.bed_edge_detector = BedEdgeDetector(
                roi=tuple(config.bed_edge_roi) if config.bed_edge_roi else None,
                color_mode=config.bed_edge_color_mode,
            )

        os.makedirs(os.path.dirname(config.output_csv), exist_ok=True)

        if config.save_debug_frames:
            os.makedirs(config.debug_dir, exist_ok=True)

        if config.save_summary:
            os.makedirs(os.path.dirname(config.summary_csv), exist_ok=True)

    def _resize_if_needed(self, frame):
        if self.config.resize_width and self.config.resize_height:
            return cv2.resize(
                frame,
                (self.config.resize_width, self.config.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _smooth_bed_edge(self, y_raw: Optional[float]) -> Optional[float]:
        if y_raw is None or np.isnan(y_raw):
            return None

        if not self.config.bed_edge_smoothing:
            return y_raw

        self.bed_edge_history.append(y_raw)
        median_y = float(np.median(list(self.bed_edge_history)))

        if self.bed_edge_ema is None:
            self.bed_edge_ema = median_y
        else:
            alpha = self.config.bed_edge_ema_alpha
            self.bed_edge_ema = alpha * median_y + (1.0 - alpha) * self.bed_edge_ema

        return float(self.bed_edge_ema)

    def _segment_with_reference(self, segmenter, frame, bed_edge_y_smooth):
        try:
            return segmenter.segment(frame, bed_edge_y=bed_edge_y_smooth)
        except TypeError:
            return segmenter.segment(frame)

    def _select_particle_bed_detection(self, detections):
        if not detections:
            return None

        candidates = []

        for det in detections:
            label = str(det.get("label", "")).lower()

            if self.config.segmentation_mode == "bed_cv":
                candidates.append(det)
            elif label in ["bed", "particle_bed", "particle bed"]:
                candidates.append(det)
            elif "bed" in label:
                candidates.append(det)

        if not candidates:
            return None

        return max(candidates, key=lambda d: d.get("area", 0.0))

    def _draw_center(self, vis, detection, color, label, radius=6):
        if detection is None:
            return

        cx, cy = detection["center"]
        cx = int(cx)
        cy = int(cy)

        cv2.circle(vis, (cx, cy), radius, color, -1)
        cv2.putText(
            vis,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _write_debug_frame(
        self,
        frame_idx: int,
        frame,
        mask,
        tracks,
        bed_edge_y_raw=None,
        bed_edge_y_smooth=None,
        inner_pipe_detections=None,
        optical_box_detections=None,
        particle_bed_detection=None,
    ) -> None:
        vis = frame.copy()

        if bed_edge_y_smooth is not None:
            y = int(bed_edge_y_smooth)
            cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (255, 0, 0), 2)
            cv2.putText(
                vis,
                f"bed_edge={y}",
                (10, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        if particle_bed_detection is not None:
            self._draw_center(
                vis,
                particle_bed_detection,
                (0, 165, 255),
                "particle_bed",
                radius=6,
            )

        if inner_pipe_detections:
            self._draw_center(
                vis,
                inner_pipe_detections[0],
                (255, 255, 0),
                "inner_pipe",
                radius=6,
            )

        if len(mask.shape) == 2:
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_vis = mask

        combined = cv2.hconcat([vis, mask_vis])
        out_path = os.path.join(self.config.debug_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, combined)

    def run(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video konnte nicht geöffnet werden: {self.config.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))

        start = self.config.start_frame
        end = self.config.end_frame if self.config.end_frame is not None else total_frames
        end = min(end, total_frames)

        if start >= end:
            raise ValueError(f"Ungültiger Framebereich: start={start}, end={end}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        processed_frames = 0
        frame_idx = start

        with open(self.config.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "frame", "time_seconds", "track_id",
                    "x", "y", "w", "h",
                    "center_x", "center_y", "area",
                    "bed_edge_y_raw", "bed_edge_y_smooth",
                    "particle_bed_x", "particle_bed_y", "particle_bed_w", "particle_bed_h",
                    "particle_bed_center_x", "particle_bed_center_y", "particle_bed_area",
                    "inner_pipe_x", "inner_pipe_y", "inner_pipe_w", "inner_pipe_h",
                    "inner_pipe_center_x", "inner_pipe_center_y",
                    "optical_box_x", "optical_box_y", "optical_box_w", "optical_box_h",
                    "optical_box_center_x", "optical_box_center_y",
                    "processing_time_s",
                ]
            )

            while frame_idx < end:
                ok, frame = cap.read()
                if not ok:
                    break

                self.evaluator.start_timer()

                frame = self._resize_if_needed(frame)

                if self.deflicker is not None:
                    frame = self.deflicker.apply(frame)

                bed_edge_y_raw = None
                bed_edge_y_smooth = None

                if self.bed_edge_detector is not None:
                    bed_edge_result = self.bed_edge_detector.detect(frame)
                    if bed_edge_result.get("success", True):
                        bed_edge_y_raw = bed_edge_result.get("y_edge")
                        bed_edge_y_smooth = self._smooth_bed_edge(bed_edge_y_raw)

                mask, detections = self._segment_with_reference(
                    self.segmenter,
                    frame,
                    bed_edge_y_smooth,
                )

                debug_mask = mask.copy()

                particle_bed_detection = self._select_particle_bed_detection(detections)

                if particle_bed_detection is not None:
                    pb_x, pb_y, pb_w, pb_h = particle_bed_detection["bbox"]
                    pb_cx, pb_cy = particle_bed_detection["center"]
                    pb_area = particle_bed_detection.get("area", np.nan)
                else:
                    pb_x = pb_y = pb_w = pb_h = ""
                    pb_cx = pb_cy = ""
                    pb_area = ""

                inner_pipe_detections = []
                if self.inner_pipe_segmenter is not None:
                    inner_pipe_mask, inner_pipe_detections = self.inner_pipe_segmenter.segment(
                        frame,
                        bed_edge_y=bed_edge_y_smooth,
                    )
                    debug_mask = cv2.bitwise_or(debug_mask, inner_pipe_mask)

                optical_box_detections = []
                if self.optical_box_segmenter is not None:
                    optical_box_mask, optical_box_detections = self.optical_box_segmenter.segment(frame)
                    debug_mask = cv2.bitwise_or(debug_mask, optical_box_mask)

                if bed_edge_y_smooth is not None:
                    cv2.line(
                        debug_mask,
                        (0, int(bed_edge_y_smooth)),
                        (debug_mask.shape[1] - 1, int(bed_edge_y_smooth)),
                        255,
                        2,
                    )

                if self.tracker is not None:
                    tracker_result = self.tracker.update(detections)

                    if isinstance(tracker_result, dict):
                        tracks = [tracker_result]
                    elif tracker_result is None:
                        tracks = []
                    else:
                        tracks = tracker_result
                else:
                    tracks = []

                elapsed = self.evaluator.stop_timer()
                self.evaluator.add_track_count(len(tracks))
                self.tracking_stats.update(tracks)

                time_seconds = frame_idx / video_fps if video_fps > 0 else 0.0

                if inner_pipe_detections:
                    ip = inner_pipe_detections[0]
                    ip_x, ip_y, ip_w, ip_h = ip["bbox"]
                    ip_cx, ip_cy = ip["center"]
                else:
                    ip_x = ip_y = ip_w = ip_h = ""
                    ip_cx = ip_cy = ""

                if optical_box_detections:
                    ob = optical_box_detections[0]
                    ob_x, ob_y, ob_w, ob_h = ob["bbox"]
                    ob_cx, ob_cy = ob["center"]
                else:
                    ob_x = ob_y = ob_w = ob_h = ""
                    ob_cx = ob_cy = ""

                rows_to_write = tracks if tracks else [None]

                for t in rows_to_write:
                    if t is not None:
                        x, y, w, h = t["bbox"]
                        cx, cy = t["center"]
                        track_id = t["track_id"]
                        area = t["area"]
                    else:
                        x = y = w = h = ""
                        cx = cy = ""
                        track_id = -1
                        area = ""

                    writer.writerow(
                        [
                            frame_idx,
                            round(time_seconds, 6),
                            track_id,
                            x, y, w, h,
                            round(cx, 3) if cx != "" else "",
                            round(cy, 3) if cy != "" else "",
                            round(area, 3) if area != "" else "",
                            round(bed_edge_y_raw, 3) if bed_edge_y_raw is not None else "",
                            round(bed_edge_y_smooth, 3) if bed_edge_y_smooth is not None else "",
                            pb_x, pb_y, pb_w, pb_h,
                            round(pb_cx, 3) if pb_cx != "" else "",
                            round(pb_cy, 3) if pb_cy != "" else "",
                            round(pb_area, 3) if pb_area != "" else "",
                            ip_x, ip_y, ip_w, ip_h,
                            round(ip_cx, 3) if ip_cx != "" else "",
                            round(ip_cy, 3) if ip_cy != "" else "",
                            ob_x, ob_y, ob_w, ob_h,
                            round(ob_cx, 3) if ob_cx != "" else "",
                            round(ob_cy, 3) if ob_cy != "" else "",
                            round(elapsed, 6),
                        ]
                    )

                if self.config.save_debug_frames:
                    self._write_debug_frame(
                        frame_idx,
                        frame,
                        debug_mask,
                        tracks,
                        bed_edge_y_raw=bed_edge_y_raw,
                        bed_edge_y_smooth=bed_edge_y_smooth,
                        inner_pipe_detections=inner_pipe_detections,
                        optical_box_detections=optical_box_detections,
                        particle_bed_detection=particle_bed_detection,
                    )

                processed_frames += 1
                frame_idx += 1

        cap.release()

        summary = merge_summaries(
            {
                "video_path": self.config.video_path,
                "output_csv": self.config.output_csv,
                "start_frame": start,
                "end_frame": end,
                "processed_frames": processed_frames,
                "video_fps": video_fps,
                "segmentation_mode": self.config.segmentation_mode,
                "device": self.config.device,
                "deflicker_enabled": self.config.enable_deflicker,
                "deflicker_freq": self.config.deflicker_freq,
                "bed_edge_smoothing": self.config.bed_edge_smoothing,
                "enable_inner_pipe": self.config.enable_inner_pipe,
                "enable_optical_box": self.config.enable_optical_box,
            },
            self.evaluator.summary(),
            self.tracking_stats.to_dict(),
        )

        if self.config.save_summary:
            save_summary_csv(summary, self.config.summary_csv)

        return summary


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PipelineConfig(
        video_path=data["video_path"],
        output_csv=data["output_csv"],
        start_frame=data.get("start_frame", 0),
        end_frame=data.get("end_frame"),
        resize_width=data.get("resize_width"),
        resize_height=data.get("resize_height"),
        enable_deflicker=data.get("enable_deflicker", True),
        enable_tracking=data.get("enable_tracking", True),
        enable_bed_edge=data.get("enable_bed_edge", True),
        enable_inner_pipe=data.get("enable_inner_pipe", False),
        enable_optical_box=data.get("enable_optical_box", True),
        save_debug_frames=data.get("save_debug_frames", False),
        debug_dir=data.get("debug_dir", "outputs/debug"),
        segmentation_mode=data.get("segmentation_mode", "pipe_cv"),
        segmentation_color_mode=data.get("segmentation_color_mode", "gray"),
        detectron2_config_file=data.get("detectron2_config_file"),
        detectron2_weights_file=data.get("detectron2_weights_file"),
        detectron2_score_threshold=data.get("detectron2_score_threshold", 0.5),
        device=data.get("device", "cpu"),
        save_summary=data.get("save_summary", True),
        summary_csv=data.get("summary_csv", "outputs/summary.csv"),
        pipe_roi=data.get("pipe_roi"),
        bed_roi=data.get("bed_roi"),
        bed_edge_roi=data.get("bed_edge_roi"),
        inner_pipe_roi=data.get("inner_pipe_roi"),
        optical_box_roi=data.get("optical_box_roi"),
        bed_edge_color_mode=data.get("bed_edge_color_mode", "gray"),
        bed_edge_smoothing=data.get("bed_edge_smoothing", True),
        bed_edge_median_window=data.get("bed_edge_median_window", 5),
        bed_edge_ema_alpha=data.get("bed_edge_ema_alpha", 0.3),
        deflicker_fps=data.get("deflicker_fps", 200.0),
        deflicker_freq=data.get("deflicker_freq", 50.0),
        deflicker_use_second_harmonic=data.get("deflicker_use_second_harmonic", False),
        deflicker_window_size=data.get("deflicker_window_size", 256),
        deflicker_min_history=data.get("deflicker_min_history", 32),
        deflicker_smooth_alpha=data.get("deflicker_smooth_alpha", 0.4),
        deflicker_use_median=data.get("deflicker_use_median", True),
        deflicker_roi=data.get("deflicker_roi"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = VideoPipeline(config)
    summary = pipeline.run()

    print("Pipeline abgeschlossen.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()