import argparse
import os

import cv2
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

    frames = list(range(args.start_frame, end_frame))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def read_gray_frame(frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Frame konnte nicht gelesen werden: {frame_idx}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    first = read_gray_frame(args.start_frame)

    fig, ax = plt.subplots(figsize=(15, 5))
    img = ax.imshow(first, cmap="gray")
    scat = ax.scatter([], [], c="r", s=args.point_size, marker=".")

    ax.set_title("Optical Flow Tracking Points")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    def update(frame_idx):
        gray = read_gray_frame(frame_idx)
        img.set_data(gray)

        frame_points = df[df["frame"] == frame_idx]

        if len(frame_points) > 0:
            xy = frame_points[["x", "y"]].values
            scat.set_offsets(xy)
        else:
            scat.set_offsets([])

        ax.set_title(f"Optical Flow Tracking Points - Frame {frame_idx}")
        return img, scat

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=args.interval,
        blit=False,
    )

    ani.save(args.out)
    cap.release()
    plt.close(fig)

    print(f"Animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()