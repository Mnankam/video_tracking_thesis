import argparse
import os

import cv2
import numpy as np
import pandas as pd


def draw_axes(vis, width, height, tick_step=200):
    axis_color = (255, 255, 255)
    text_color = (255, 255, 255)

    # Hintergrund leicht abdunkeln für bessere Lesbarkeit
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (width, 70), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 45), (width, height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 0), (80, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # x-Achse unten
    y_axis_pos = height - 35
    cv2.line(vis, (80, y_axis_pos), (width - 20, y_axis_pos), axis_color, 1)

    for x in range(0, width + 1, tick_step):
        if x < 80:
            continue

        cv2.line(
            vis,
            (x, y_axis_pos - 6),
            (x, y_axis_pos + 6),
            axis_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            str(x),
            (x - 20, y_axis_pos + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        "x [px]",
        (width // 2 - 40, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        1,
        cv2.LINE_AA,
    )

    # y-Achse links
    x_axis_pos = 70
    cv2.line(vis, (x_axis_pos, 20), (x_axis_pos, height - 45), axis_color, 1)

    for y in range(0, height + 1, tick_step):
        if y > height - 45:
            continue

        cv2.line(
            vis,
            (x_axis_pos - 6, y),
            (x_axis_pos + 6, y),
            axis_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            str(y),
            (10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        "y [px]",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="Fast OpenCV animation for Optical Flow")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--point-radius", type=int, default=5)
    parser.add_argument("--trail-length", type=int, default=20)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--tick-step", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {"frame", "point_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}")

    if "tracking_status" in df.columns:
        df = df[df["tracking_status"] == 1].copy()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = max(0, int(args.start_frame))
    end_frame = args.end_frame if args.end_frame is not None else total_frames
    end_frame = min(int(end_frame), total_frames)

    if start_frame >= end_frame:
        raise ValueError("Ungültiger Framebereich.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Start-Frame konnte nicht gelesen werden: {start_frame}")

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
                print(
                    f"Warnung: Frame {frame_idx} konnte nicht gelesen werden. "
                    "Stoppe Animation."
                )
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        draw_axes(vis, width, height, tick_step=args.tick_step)

        # Bewegungsspur
        history = df[
            (df["frame"] <= frame_idx)
            & (df["frame"] > frame_idx - args.trail_length)
        ]

        for pid in sorted(history["point_id"].unique()):
            h = history[history["point_id"] == pid].sort_values("frame")
            pts = h[["x", "y"]].values.astype(int)

            for i in range(1, len(pts)):
                cv2.line(
                    vis,
                    tuple(pts[i - 1]),
                    tuple(pts[i]),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # aktuelle Punkte
        frame_points = df[df["frame"] == frame_idx]

        for _, row in frame_points.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            pid = int(row["point_id"])

            cv2.circle(vis, (x, y), args.point_radius, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"id={pid}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Optical Flow Tracking Points - Frame {frame_idx}",
            (90, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"OpenCV animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()