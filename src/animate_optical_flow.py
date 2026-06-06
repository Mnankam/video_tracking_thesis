import argparse
import os

import cv2
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Animate Optical Flow points")
    parser.add_argument("--video", required=True, help="Pfad zum Video")
    parser.add_argument("--csv", required=True, help="Optical-Flow-CSV")
    parser.add_argument("--out", required=True, help="Ausgabe-MP4")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--point-radius", type=int, default=6)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = {"frame", "point_id", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}")

    if "tracking_status" in df.columns:
        df = df[df["tracking_status"] == 1].copy()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = args.end_frame if args.end_frame is not None else total
    end_frame = min(end_frame, total)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Erster Frame konnte nicht gelesen werden.")

    h, w = frame.shape[:2]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (w, h),
    )

    frame_idx = args.start_frame

    while frame_idx < end_frame:
        if frame_idx != args.start_frame:
            ok, frame = cap.read()
            if not ok:
                break

        # aktuelle Punkte
        frame_points = df[df["frame"] == frame_idx]

        # Bewegungsspur pro Punkt
        history = df[(df["frame"] <= frame_idx) & (df["frame"] > frame_idx - args.trail_length)]

        for pid in sorted(history["point_id"].unique()):
            hdf = history[history["point_id"] == pid]
            pts = hdf[["x", "y"]].values.astype(int)

            for j in range(1, len(pts)):
                cv2.line(
                    frame,
                    tuple(pts[j - 1]),
                    tuple(pts[j]),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        for _, row in frame_points.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            point_id = int(row["point_id"])

            cv2.circle(frame, (x, y), args.point_radius, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"id={point_id}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()