from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import yaml

from src.deflicker import FFTDeflicker
from src.segmentation import PipeSegmenterCV, BedSegmenterCV, Detectron2Segmenter
from src.bed_edge import BedEdgeDetector
from src.tracking import SingleObjectTracker, MultiObjectTracker
from src.evaluation import (
    Evaluator,
    TrackingStats,
    merge_summaries,
    save_summary_csv,
    evaluate_all_metrics,
    
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

    save_debug_frames: bool = False
    debug_dir: str = "outputs/debug"

    segmentation_mode: str = "pipe_cv"
    detectron2_config_file: Optional[str] = None
    detectron2_weights_file: Optional[str] = None
    detectron2_score_threshold: float = 0.5
    device: str = "cpu"

    save_summary: bool = True
    summary_csv: str = "outputs/summary.csv"

    bed_edge_roi: Optional[list[int]] = None

    # ROI für klassische Segmentierung
    pipe_roi: Optional[list[int]] = None
    bed_roi: Optional[list[int]] = None

    # FFT-Deflicker Konfiguration
    deflicker_fps: float = 200.0
    deflicker_freq: float = 50.0
    deflicker_use_second_harmonic: bool = False
    deflicker_window_size: int = 256
    deflicker_min_history: int = 32
    deflicker_smooth_alpha: float = 0.2
    deflicker_use_median: bool = True
    deflicker_roi: Optional[list[int]] = None


class VideoPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.evaluator = Evaluator()
        self.tracking_stats = TrackingStats()

        # FFT-Deflicker
        self.deflicker = None
        if config.enable_deflicker:
            deflicker_roi_tuple = (
                tuple(config.deflicker_roi) if config.deflicker_roi else None
            )
            self.deflicker = FFTDeflicker(
                fps=config.deflicker_fps,
                flicker_freq=config.deflicker_freq,
                use_second_harmonic=config.deflicker_use_second_harmonic,
                window_size=config.deflicker_window_size,
                min_history=config.deflicker_min_history,
                smooth_alpha=config.deflicker_smooth_alpha,
                use_median=config.deflicker_use_median,
                roi=deflicker_roi_tuple,
            )

        # Segmentierung
        if config.segmentation_mode == "detectron2":
            if not config.detectron2_config_file:
                raise ValueError(
                    "detectron2_config_file fehlt, obwohl segmentation_mode='detectron2' gesetzt ist."
                )
            if not config.detectron2_weights_file:
                raise ValueError(
                    "detectron2_weights_file fehlt, obwohl segmentation_mode='detectron2' gesetzt ist."
                )

            self.segmenter = Detectron2Segmenter(
                config_file=config.detectron2_config_file,
                weights_file=config.detectron2_weights_file,
                score_threshold=config.detectron2_score_threshold,
                device=config.device,
            )

        elif config.segmentation_mode == "pipe_cv":
            pipe_roi_tuple = tuple(config.pipe_roi) if config.pipe_roi else None
            self.segmenter = PipeSegmenterCV(roi=pipe_roi_tuple)

        elif config.segmentation_mode == "bed_cv":
            bed_roi_tuple = tuple(config.bed_roi) if config.bed_roi else None
            self.segmenter = BedSegmenterCV(roi=bed_roi_tuple)

        else:
            raise ValueError(
                f"Unbekannter segmentation_mode: {config.segmentation_mode}. "
                f"Erwartet: 'pipe_cv', 'bed_cv' oder 'detectron2'."
            )

        # Tracking
        if config.enable_tracking:
            if config.segmentation_mode == "pipe_cv":
                self.tracker = SingleObjectTracker()
            else:
                self.tracker = MultiObjectTracker(max_distance=60.0, max_missed=5)
        else:
            self.tracker = None

        # Bettkante
        self.bed_edge_detector = None
        if config.enable_bed_edge:
            roi_tuple = tuple(config.bed_edge_roi) if config.bed_edge_roi else None
            self.bed_edge_detector = BedEdgeDetector(roi=roi_tuple)

        output_dir = os.path.dirname(config.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if config.save_debug_frames:
            os.makedirs(config.debug_dir, exist_ok=True)

        summary_dir = os.path.dirname(config.summary_csv)
        if config.save_summary and summary_dir:
            os.makedirs(summary_dir, exist_ok=True)

    def _resize_if_needed(self, frame):
        if self.config.resize_width and self.config.resize_height:
            return cv2.resize(
                frame,
                (self.config.resize_width, self.config.resize_height),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _write_debug_frame(
        self, frame_idx: int, frame, mask, tracks, bed_edge_y=None
    ) -> None:
        vis = frame.copy()

        for track in tracks:
            x, y, w, h = track["bbox"]
            cx, cy = track["center"]

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"id={track['track_id']}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if bed_edge_y is not None:
            cv2.line(
                vis,
                (0, int(bed_edge_y)),
                (vis.shape[1] - 1, int(bed_edge_y)),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                vis,
                f"bed_edge_y={int(bed_edge_y)}",
                (10, max(25, int(bed_edge_y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
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
                    "frame",
                    "time_seconds",
                    "track_id",
                    "x",
                    "y",
                    "w",
                    "h",
                    "center_x",
                    "center_y",
                    "area",
                    "bed_edge_y",
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

                bed_edge_y = None
                if self.bed_edge_detector is not None:
                    bed_edge_result = self.bed_edge_detector.detect(frame)
                    bed_edge_y = bed_edge_result["y_edge"]

                mask, detections = self.segmenter.segment(frame)

                if self.tracker is not None:
                    tracker_result = self.tracker.update(detections)

                    # SingleObjectTracker gibt entweder dict oder None zurück
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

                if tracks:
                    for t in tracks:
                        x, y, w, h = t["bbox"]
                        cx, cy = t["center"]

                        writer.writerow(
                            [
                                frame_idx,
                                round(time_seconds, 6),
                                t["track_id"],
                                x,
                                y,
                                w,
                                h,
                                round(cx, 3),
                                round(cy, 3),
                                round(t["area"], 3),
                                round(bed_edge_y, 3) if bed_edge_y is not None else "",
                                round(elapsed, 6),
                            ]
                        )
                else:
                    writer.writerow(
                        [
                            frame_idx,
                            round(time_seconds, 6),
                            -1,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            round(bed_edge_y, 3) if bed_edge_y is not None else "",
                            round(elapsed, 6),
                        ]
                    )

                if self.config.save_debug_frames:
                    self._write_debug_frame(frame_idx, frame, mask, tracks, bed_edge_y)

                processed_frames += 1
                frame_idx += 1

        cap.release()

        pipeline_summary = {
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
        }

        eval_summary = self.evaluator.summary()
        tracking_summary = self.tracking_stats.to_dict()

        summary = merge_summaries(pipeline_summary, eval_summary, tracking_summary)

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
        save_debug_frames=data.get("save_debug_frames", False),
        debug_dir=data.get("debug_dir", "outputs/debug"),
        segmentation_mode=data.get("segmentation_mode", "pipe_cv"),
        detectron2_config_file=data.get("detectron2_config_file"),
        detectron2_weights_file=data.get("detectron2_weights_file"),
        detectron2_score_threshold=data.get("detectron2_score_threshold", 0.5),
        device=data.get("device", "cpu"),
        save_summary=data.get("save_summary", True),
        summary_csv=data.get("summary_csv", "outputs/summary.csv"),
        bed_edge_roi=data.get("bed_edge_roi"),
        pipe_roi=data.get("pipe_roi"),
        bed_roi=data.get("bed_roi"),
        deflicker_fps=data.get("deflicker_fps", 200.0),
        deflicker_freq=data.get("deflicker_freq", 50.0),
        deflicker_use_second_harmonic=data.get("deflicker_use_second_harmonic", False),
        deflicker_window_size=data.get("deflicker_window_size", 256),
        deflicker_min_history=data.get("deflicker_min_history", 32),
        deflicker_smooth_alpha=data.get("deflicker_smooth_alpha", 0.2),
        deflicker_use_median=data.get("deflicker_use_median", True),
        deflicker_roi=data.get("deflicker_roi"),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Pfad zur YAML-Konfigurationsdatei",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = VideoPipeline(config)
    summary = pipeline.run()

    print("Pipeline abgeschlossen.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()