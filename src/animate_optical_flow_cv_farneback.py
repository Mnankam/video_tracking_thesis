import argparse
import os

import cv2
import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resize_frame(frame, config):
    rw = config.get("resize_width")
    rh = config.get("resize_height")

    if rw is not None and rh is not None:
        return cv2.resize(frame, (int(rw), int(rh)), interpolation=cv2.INTER_AREA)

    return frame


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    return gray


def crop_roi(flow, roi):
    if roi is None:
        return flow, 0, 0

    x, y, w, h = map(int, roi)
    H, W = flow.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return flow[y:y + h, x:x + w], x, y


def draw_flow_arrows(vis, flow, roi, step=25, scale=8.0):
    roi_flow, x0, y0 = crop_roi(flow, roi)
    h, w = roi_flow.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = roi_flow[y, x]

            x_start = int(x0 + x)
            y_start = int(y0 + y)
            x_end = int(x_start + scale * dx)
            y_end = int(y_start + scale * dy)

            mag = np.sqrt(dx * dx + dy * dy)

            if mag < 0.05:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.arrowedLine(
                vis,
                (x_start, y_start),
                (x_end, y_end),
                color,
                1,
                cv2.LINE_AA,
                tipLength=0.3,
            )


def main():
    parser = argparse.ArgumentParser(description="Animate Farneback dense optical flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--scale", type=float, default=8.0)
    args = parser.parse_args()

    config = load_config(args.config)

    video_path = config["video_path"]
    roi = config.get("inner_pipe_roi")

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

    ok, prev_frame = cap.read()
    if not ok:
        raise RuntimeError("Startframe konnte nicht gelesen werden.")

    prev_frame = resize_frame(prev_frame, config)
    prev_gray = preprocess(prev_frame)

    height, width = prev_frame.shape[:2]
    out_fps = args.fps if args.fps is not None else (video_fps if video_fps > 0 else 30.0)

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"MJPG"),
        out_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter konnte nicht geöffnet werden: {args.out}")

    frame_idx = start_frame + 1

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, config)
        gray = preprocess(frame)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            25,
            3,
            7,
            1.5,
            0,
        )

        vis = frame.copy()

        if roi is not None:
            x, y, w, h = map(int, roi)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(
                vis,
                "inner_pipe_roi",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        draw_flow_arrows(
            vis,
            flow,
            roi,
            step=args.step,
            scale=args.scale,
        )

        roi_flow, _, _ = crop_roi(flow, roi)
        dx = roi_flow[:, :, 0]
        dy = roi_flow[:, :, 1]
        mag = np.sqrt(dx ** 2 + dy ** 2)

        text = (
            f"Farneback Dense Flow | Frame {frame_idx} | "
            f"mean_dx={np.mean(dx):.3f}, mean_dy={np.mean(dy):.3f}, "
            f"mean_mag={np.mean(mag):.3f}"
        )

        cv2.putText(
            vis,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

        prev_gray = gray.copy()
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Farneback Animation gespeichert: {args.out}")


if __name__ == "__main__":
    main()