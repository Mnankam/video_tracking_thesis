import argparse
import os

import cv2
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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

        frame_points = df[df["frame"] == frame_idx]

        for _, row in frame_points.iterrows():
            x = int(row["x"])
            y = int(row["y"])
            point_id = int(row["point_id"])

            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"id={point_id}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()
