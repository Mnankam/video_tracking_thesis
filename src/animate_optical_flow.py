import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    parser = argparse.ArgumentParser(description="Matplotlib animation for Optical Flow")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--point-size", type=int, default=25)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--bitrate", type=int, default=1800)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "tracking_status" in df.columns:
        df = df[df["tracking_status"] == 1].copy()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = args.end_frame if args.end_frame is not None else total
    end_frame = min(end_frame, total)

    if args.start_frame >= end_frame:
        raise ValueError("Keine Frames für Animation ausgewählt.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Start-Frame konnte nicht gelesen werden: {args.start_frame}")

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    fig, ax = plt.subplots(figsize=(15, 5))
    img = ax.imshow(first_gray, cmap="gray")
    scat = ax.scatter([], [], c="r", s=args.point_size, marker=".")

    ax.set_title(f"Optical Flow Tracking Points - Frame {args.start_frame}")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    current_frame_idx = args.start_frame

    def update(_):
        nonlocal current_frame_idx

        if current_frame_idx == args.start_frame:
            gray = first_gray
        else:
            ok, frame = cap.read()
            if not ok:
                gray = np.zeros_like(first_gray)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img.set_data(gray)

        frame_points = df[df["frame"] == current_frame_idx]

        if len(frame_points) > 0:
            xy = frame_points[["x", "y"]].values
            scat.set_offsets(xy)
        else:
            scat.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Optical Flow Tracking Points - Frame {current_frame_idx}")

        current_frame_idx += 1

        return img, scat

    frame_count = end_frame - args.start_frame

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=args.interval,
        blit=False,
    )

    if args.out.lower().endswith(".gif"):
        ani.save(args.out, writer="pillow")
    else:
        writer = animation.FFMpegWriter(
            fps=args.fps,
            codec="mpeg4",
            bitrate=args.bitrate,
        )
        ani.save(args.out, writer=writer)

    cap.release()
    plt.close(fig)

    print(f"Animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()