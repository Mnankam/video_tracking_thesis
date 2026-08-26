#!/usr/bin/env python3
"""
Create a frame-by-frame animation of Detectron2 inference results.

The script does not run Detectron2 inference itself. It reads the source
video sequentially and overlays detections stored in a CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    """Load the YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        )

    return config


def resize_frame(
    frame,
    config: dict,
):
    """Resize frame according to the configured output resolution."""
    width = config.get("resize_width")
    height = config.get("resize_height")

    if width is not None and height is not None:
        return cv2.resize(
            frame,
            (int(width), int(height)),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def draw_reference_rois(
    image,
    config: dict,
) -> None:
    """Draw configured reference ROIs."""

    rois = {
        "inner_pipe_roi": config.get("inner_pipe_roi"),
        "particle_bed_roi": config.get("bed_roi"),
        "bed_edge_roi": config.get("bed_edge_roi"),
    }

    colors = {
        "inner_pipe_roi": (255, 255, 0),
        "particle_bed_roi": (0, 165, 255),
        "bed_edge_roi": (255, 0, 0),
    }

    for name, roi in rois.items():
        if roi is None:
            continue

        x, y, width, height = map(int, roi)
        color = colors[name]

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            2,
        )

        cv2.putText(
            image,
            name,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def build_detection_lookup(
    detections: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Group detections by frame number for efficient lookup."""

    detections = detections.copy()

    detections["frame"] = pd.to_numeric(
        detections["frame"],
        errors="raise",
    ).astype(int)

    return {
        int(frame_number): group
        for frame_number, group in detections.groupby(
            "frame",
            sort=False,
        )
    }


def print_csv_frame_diagnostics(
    detections_by_frame: dict[int, pd.DataFrame],
    expected_stride: int,
) -> None:
    """Print information about the frame coverage in the inference CSV."""

    frame_numbers = sorted(detections_by_frame)

    if not frame_numbers:
        print(
            "WARNING: The CSV contains no detection rows."
        )
        return

    differences = [
        current - previous
        for previous, current in zip(
            frame_numbers,
            frame_numbers[1:],
        )
    ]

    unique_differences = sorted(set(differences))

    print("=" * 72)
    print("CSV frame coverage")
    print("=" * 72)
    print(f"First frame with detection : {frame_numbers[0]}")
    print(f"Last frame with detection  : {frame_numbers[-1]}")
    print(
        f"Frames with detections     : "
        f"{len(frame_numbers)}"
    )
    print(
        f"Observed frame differences : "
        f"{unique_differences[:20]}"
    )
    print(f"Expected inference stride  : {expected_stride}")

    if differences and any(
        difference > expected_stride
        for difference in differences
    ):
        print(
            "WARNING: The detection CSV has gaps larger "
            "than the expected stride. This usually means "
            "that inference skipped frames or that frames "
            "without detections are not written to the CSV."
        )

    print("=" * 72)


def draw_detections(
    image,
    detections: pd.DataFrame,
) -> int:
    """Draw all detections for one frame and return their count."""

    if detections is None or detections.empty:
        return 0

    count = 0

    for _, row in detections.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        width = int(row["w"])
        height = int(row["h"])

        score = row.get("score")
        class_id = row.get("class_id")

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        label = "detectron2"

        if pd.notna(class_id):
            label += f" id={int(class_id)}"

        if pd.notna(score):
            label += f" score={float(score):.2f}"

        cv2.putText(
            image,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        count += 1

    return count


def main() -> None:
    """Create the video animation."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a frame-by-frame Detectron2 "
            "visualization from a video and CSV."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--expected-stride",
        type=int,
        default=1,
        help=(
            "Expected inference stride. Use 1 when "
            "every input frame should be processed."
        ),
    )

    parser.add_argument(
        "--draw-reference-rois",
        action="store_true",
    )

    args = parser.parse_args()

    if args.expected_stride < 1:
        raise ValueError(
            "--expected-stride must be at least 1."
        )

    config = load_config(args.config)

    video_path = Path(config["video_path"])

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not args.csv.is_file():
        raise FileNotFoundError(
            f"Detection CSV not found: {args.csv}"
        )

    detections = pd.read_csv(args.csv)

    required_columns = {
        "frame",
        "x",
        "y",
        "w",
        "h",
    }

    missing_columns = required_columns - set(
        detections.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in Detectron2 CSV: "
            f"{sorted(missing_columns)}"
        )

    detections_by_frame = build_detection_lookup(
        detections
    )

    print_csv_frame_diagnostics(
        detections_by_frame,
        args.expected_stride,
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Video could not be opened: {video_path}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    start_frame = max(0, args.start_frame)

    end_frame = (
        args.end_frame
        if args.end_frame is not None
        else total_frames
    )

    end_frame = min(end_frame, total_frames)

    if start_frame >= end_frame:
        raise ValueError(
            "Invalid frame range: "
            f"{start_frame} to {end_frame}."
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    success, first_frame = capture.read()

    if not success:
        raise RuntimeError(
            "The requested start frame could not be read."
        )

    first_frame = resize_frame(
        first_frame,
        config,
    )

    height, width = first_frame.shape[:2]

    output_fps = (
        args.fps
        if args.fps is not None
        else video_fps
    )

    if output_fps <= 0:
        output_fps = 30.0

    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter could not be opened: {args.out}"
        )

    frame_idx = start_frame
    frame = first_frame

    try:
        while frame_idx < end_frame:
            if frame_idx != start_frame:
                success, frame = capture.read()

                if not success:
                    print(
                        "WARNING: Video ended before "
                        "the configured end frame."
                    )
                    break

                frame = resize_frame(
                    frame,
                    config,
                )

            visualization = frame.copy()

            if args.draw_reference_rois:
                draw_reference_rois(
                    visualization,
                    config,
                )

            frame_detections = detections_by_frame.get(
                frame_idx
            )

            num_detections = draw_detections(
                visualization,
                frame_detections,
            )

            status = (
                "detections available"
                if frame_detections is not None
                else "no CSV entry"
            )

            cv2.putText(
                visualization,
                f"Detectron2 GPU Inference - "
                f"Frame {frame_idx}",
                (30, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                visualization,
                f"CSV status: {status} | "
                f"boxes: {num_detections}",
                (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(visualization)

            frame_idx += 1

    finally:
        capture.release()
        writer.release()

    print(
        "Detectron2 animation saved:\n"
        f"  {args.out}"
    )


if __name__ == "__main__":
    main()