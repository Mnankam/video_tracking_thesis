import argparse
import os

import cv2
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resize_frame(frame, config):
    rw = config.get("resize_width")
    rh = config.get("resize_height")

    if rw is not None and rh is not None:
        return cv2.resize(
            frame,
            (int(rw), int(rh)),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def draw_reference_rois(vis, config):
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

        x, y, w, h = map(int, roi)
        color = colors[name]

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis,
            name,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Animate Detectron2 results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--draw-reference-rois", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = config["video_path"]

    df = pd.read_csv(args.csv)

    required = {"frame", "x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in Detectron2 CSV: {missing}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))

    start_frame = max(0, int(args.start_frame))
    end_frame = args.end_frame if args.end_frame is not None else total_frames
    end_frame = min(int(end_frame), total_frames)

    if start_frame >= end_frame:
        raise ValueError("Ungültiger Framebereich.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Startframe konnte nicht gelesen werden.")

    frame = resize_frame(frame, config)
    height, width = frame.shape[:2]

    out_fps = args.fps if args.fps is not None else (video_fps if video_fps > 0 else 30.0)

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"MJPG"),
        out_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter konnte nicht geöffnet werden: {args.out}")

    frame_idx = start_frame

    while frame_idx < end_frame:
        if frame_idx != start_frame:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_frame(frame, config)

        vis = frame.copy()

        if args.draw_reference_rois:
            draw_reference_rois(vis, config)

        frame_detections = df[df["frame"] == frame_idx]

        for _, row in frame_detections.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])

            score = row["score"] if "score" in row else None
            class_id = row["class_id"] if "class_id" in row else None

            cv2.rectangle(
                vis,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            label = "detectron2"

            if class_id is not None:
                label += f" id={int(class_id)}"

            if score is not None:
                label += f" score={float(score):.2f}"

            cv2.putText(
                vis,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Detectron2 GPU Inference - Frame {frame_idx}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Detectron2 animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()