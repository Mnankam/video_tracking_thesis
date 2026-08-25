#!/usr/bin/env python3
"""
Detectron2 inference for complete video sequences.

For every processed video this script creates three outputs:

1. <video>_detectron2_results.csv
   Object-level Detectron2 results. One row per detection.

2. <video>_detectron2_results_frames.csv
   Frame-level processing results. Exactly one row per successfully
   decoded and processed frame, including frames without detections.

3. <video>_detectron2_results_benchmark.csv
   Video-level summary containing processing coverage, detection-output
   rate, runtime, and throughput.

The separate frame-level file is important because the object-level
Detectron2 CSV contains no row for frames without detections and can
contain multiple rows for frames with several detections.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml


# =============================================================================
# Configuration
# =============================================================================


def load_config(path: str) -> dict:
    """Load YAML configuration."""

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    return config


def resize_frame(frame, config):
    """Resize a frame if both target dimensions are configured."""

    resize_width = config.get("resize_width")
    resize_height = config.get("resize_height")

    if resize_width is not None and resize_height is not None:
        frame = cv2.resize(
            frame,
            (int(resize_width), int(resize_height)),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def draw_roi_boxes(image, config):
    """Draw configured experiment ROIs on debug images."""

    rois = {
        "inner_pipe_roi": config.get("inner_pipe_roi"),
        "bed_roi": config.get("bed_roi"),
        "bed_edge_roi": config.get("bed_edge_roi"),
    }

    colors = {
        "inner_pipe_roi": (255, 255, 0),
        "bed_roi": (0, 165, 255),
        "bed_edge_roi": (255, 0, 0),
    }

    for name, roi in rois.items():
        if roi is None:
            continue

        x, y, width, height = map(int, roi)

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            colors[name],
            2,
        )


# =============================================================================
# Detectron2
# =============================================================================


def build_predictor(config):
    """Construct the Detectron2 DefaultPredictor."""

    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()

    config_file = config.get(
        "detectron2_config_file",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    )

    weights_file = config.get(
        "detectron2_weights_file",
        "COCO",
    )

    score_threshold = float(
        config.get(
            "detectron2_score_threshold",
            0.5,
        )
    )

    cfg.merge_from_file(
        model_zoo.get_config_file(config_file)
    )

    if weights_file == "COCO":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            config_file
        )
    else:
        cfg.MODEL.WEIGHTS = weights_file

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cuda"

    return DefaultPredictor(cfg)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Detectron2 inference over a complete video and create "
            "object-level, frame-level, and benchmark outputs."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--output-csv",
        default=None,
        help="Object-level Detectron2 output CSV.",
    )

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Directory for optional debug frames.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    config = load_config(args.config)

    video_path = config["video_path"]

    output_csv = (
        args.output_csv
        or config.get("detectron2_output_csv")
    )

    debug_dir = (
        args.debug_dir
        or config.get("detectron2_debug_dir")
    )

    if output_csv is None:
        raise ValueError(
            "No Detectron2 output CSV was specified."
        )

    if debug_dir is None:
        debug_dir = os.path.join(
            os.path.dirname(output_csv),
            "debug",
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(debug_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    # Debug images are intentionally sampled. This has NO effect on inference.
    debug_stride = int(
        config.get(
            "detectron2_debug_stride",
            10,
        )
    )

    if debug_stride < 1:
        debug_stride = 1

    # -------------------------------------------------------------------------
    # Predictor
    # -------------------------------------------------------------------------

    print("=" * 80)
    print("DETECTRON2 FULL-VIDEO INFERENCE")
    print("=" * 80)
    print(f"Video       : {video_path}")
    print(f"Output CSV  : {output_csv}")
    print(f"Debug dir   : {debug_dir}")

    predictor = build_predictor(config)

    # -------------------------------------------------------------------------
    # Video
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Video could not be opened: {video_path}"
        )

    total_source_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps_video = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if total_source_frames <= 0:
        cap.release()
        raise RuntimeError(
            "OpenCV returned an invalid source frame count."
        )

    if fps_video <= 0:
        cap.release()
        raise RuntimeError(
            "OpenCV returned an invalid video frame rate."
        )

    # -------------------------------------------------------------------------
    # Processing range
    # -------------------------------------------------------------------------

    start_frame = int(
        config.get(
            "start_frame",
            0,
        )
    )

    configured_end_frame = config.get(
        "end_frame",
        None,
    )

    if configured_end_frame in (
        None,
        "",
        "null",
        "None",
    ):
        end_frame = total_source_frames
    else:
        end_frame = min(
            int(configured_end_frame),
            total_source_frames,
        )

    if start_frame < 0:
        raise ValueError(
            "start_frame must be >= 0."
        )

    if start_frame >= total_source_frames:
        raise ValueError(
            "start_frame is outside the video."
        )

    if end_frame <= start_frame:
        raise ValueError(
            "end_frame must be greater than start_frame."
        )

    expected_processed_frames = (
        end_frame - start_frame
    )

    print(f"Source frames: {total_source_frames}")
    print(f"Video FPS    : {fps_video:.6f}")
    print(f"Start frame  : {start_frame}")
    print(f"End frame    : {end_frame}")
    print(
        "Expected frames in selected range: "
        f"{expected_processed_frames}"
    )
    print(
        "Debug image stride: "
        f"{debug_stride} "
        "(debug output only; inference stride is always 1)"
    )
    print("=" * 80)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    # -------------------------------------------------------------------------
    # Output containers
    # -------------------------------------------------------------------------

    detection_rows = []
    frame_rows = []

    processed_frames = 0
    decoded_frames = 0

    total_start = time.perf_counter()

    frame_idx = start_frame

    # -------------------------------------------------------------------------
    # Complete frame-by-frame inference
    # -------------------------------------------------------------------------

    while frame_idx < end_frame:

        ok, frame = cap.read()

        if not ok:
            print(
                "WARNING: Video decoding stopped at "
                f"frame {frame_idx}."
            )
            break

        decoded_frames += 1

        frame = resize_frame(
            frame,
            config,
        )

        frame_start = time.perf_counter()

        outputs = predictor(frame)

        frame_runtime = (
            time.perf_counter()
            - frame_start
        )

        instances = outputs["instances"].to("cpu")

        num_detections = len(instances)

        # ---------------------------------------------------------------------
        # One row for EVERY processed frame
        # ---------------------------------------------------------------------

        frame_rows.append(
            {
                "method": "detectron2_gpu",
                "frame": frame_idx,
                "video_time_seconds": (
                    frame_idx / fps_video
                ),
                "processed": 1,
                "num_detections": int(
                    num_detections
                ),
                "has_detection": int(
                    num_detections > 0
                ),
                "compute_time_s": frame_runtime,
            }
        )

        # ---------------------------------------------------------------------
        # Object-level detections
        # ---------------------------------------------------------------------

        if instances.has("pred_boxes"):

            boxes = (
                instances.pred_boxes
                .tensor
                .numpy()
            )

            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            for index, box in enumerate(boxes):

                x1, y1, x2, y2 = box.astype(int)

                width = int(x2 - x1)
                height = int(y2 - y1)

                detection_rows.append(
                    {
                        "method": "detectron2_gpu",
                        "frame": frame_idx,
                        "video_time_seconds": (
                            frame_idx / fps_video
                        ),
                        "class_id": int(
                            classes[index]
                        ),
                        "score": float(
                            scores[index]
                        ),
                        "x": int(x1),
                        "y": int(y1),
                        "w": width,
                        "h": height,
                        "area": int(
                            width * height
                        ),
                        "compute_time_s": (
                            frame_runtime
                        ),
                    }
                )

        # ---------------------------------------------------------------------
        # Debug image
        #
        # Only every Nth debug image is written to disk.
        # Every video frame is nevertheless passed through Detectron2.
        # ---------------------------------------------------------------------

        if frame_idx % debug_stride == 0:

            debug_image = frame.copy()

            draw_roi_boxes(
                debug_image,
                config,
            )

            cv2.imwrite(
                os.path.join(
                    debug_dir,
                    f"frame_{frame_idx:06d}.png",
                ),
                debug_image,
            )

        processed_frames += 1

        if (
            processed_frames == 1
            or processed_frames % 1000 == 0
        ):
            print(
                f"Processed {processed_frames}/"
                f"{expected_processed_frames} frames "
                f"(current source frame: {frame_idx})"
            )

        # IMPORTANT:
        # Increment by exactly one.
        frame_idx += 1

    # -------------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------------

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    cap.release()

    detection_df = pd.DataFrame(
        detection_rows
    )

    frame_df = pd.DataFrame(
        frame_rows
    )

    # -------------------------------------------------------------------------
    # Save object-level CSV
    # -------------------------------------------------------------------------

    detection_columns = [
        "method",
        "frame",
        "video_time_seconds",
        "class_id",
        "score",
        "x",
        "y",
        "w",
        "h",
        "area",
        "compute_time_s",
    ]

    if detection_df.empty:
        detection_df = pd.DataFrame(
            columns=detection_columns
        )

    detection_df.to_csv(
        output_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save frame-level CSV
    # -------------------------------------------------------------------------

    frame_output_path = (
        output_path.with_name(
            f"{output_path.stem}_frames.csv"
        )
    )

    frame_df.to_csv(
        frame_output_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    if processed_frames > 0:

        avg_frame_time = (
            total_runtime
            / processed_frames
        )

        effective_fps = (
            processed_frames
            / total_runtime
        )

        frames_with_detection = int(
            frame_df["has_detection"].sum()
        )

        frames_without_detection = int(
            processed_frames
            - frames_with_detection
        )

        processing_coverage_percent = (
            100.0
            * processed_frames
            / expected_processed_frames
        )

        detection_output_rate_percent = (
            100.0
            * frames_with_detection
            / processed_frames
        )

    else:

        avg_frame_time = float("nan")
        effective_fps = float("nan")
        frames_with_detection = 0
        frames_without_detection = 0
        processing_coverage_percent = 0.0
        detection_output_rate_percent = float("nan")

    full_length_run = int(
        start_frame == 0
        and processed_frames == total_source_frames
        and end_frame == total_source_frames
    )

    # -------------------------------------------------------------------------
    # Benchmark summary
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "method": "detectron2_gpu",
                "total_source_frames": (
                    total_source_frames
                ),
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "expected_processed_frames": (
                    expected_processed_frames
                ),
                "decoded_frames": decoded_frames,
                "processed_frames": processed_frames,
                "frames_with_detection": (
                    frames_with_detection
                ),
                "frames_without_detection": (
                    frames_without_detection
                ),
                "processing_coverage_percent": (
                    processing_coverage_percent
                ),
                "detection_output_rate_percent": (
                    detection_output_rate_percent
                ),
                "num_detections": len(
                    detection_df
                ),
                "full_length_run": (
                    full_length_run
                ),
                "total_runtime_s": (
                    total_runtime
                ),
                "avg_frame_time_s": (
                    avg_frame_time
                ),
                "effective_fps": (
                    effective_fps
                ),
            }
        ]
    )

    summary_path = (
        output_path.with_name(
            f"{output_path.stem}_benchmark.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("DETECTRON2 FULL-VIDEO SUMMARY")
    print("=" * 80)

    print(
        f"Total source frames          : "
        f"{total_source_frames}"
    )

    print(
        f"Expected processed frames    : "
        f"{expected_processed_frames}"
    )

    print(
        f"Actually processed frames    : "
        f"{processed_frames}"
    )

    print(
        f"Frames with >= 1 detection   : "
        f"{frames_with_detection}"
    )

    print(
        f"Frames without detections     : "
        f"{frames_without_detection}"
    )

    print(
        f"Processing coverage [%]       : "
        f"{processing_coverage_percent:.6f}"
    )

    print(
        f"Detection-output rate [%]     : "
        f"{detection_output_rate_percent:.6f}"
    )

    print(
        f"Number of detections          : "
        f"{len(detection_df)}"
    )

    print(
        f"Full-length run               : "
        f"{bool(full_length_run)}"
    )

    print(
        f"Total runtime [s]             : "
        f"{total_runtime:.6f}"
    )

    print(
        f"Average frame time [s]        : "
        f"{avg_frame_time:.6f}"
    )

    print(
        f"Effective FPS                 : "
        f"{effective_fps:.6f}"
    )

    print()
    print(f"Detection CSV : {output_csv}")
    print(f"Frame CSV     : {frame_output_path}")
    print(f"Benchmark CSV : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()