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
        return None, 0, 0

    x, y, w, h = map(int, roi)
    H, W = flow.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return flow[y:y + h, x:x + w], x, y


def draw_flow_arrows(vis, flow, roi, color, step=25, scale=8.0):
    roi_flow, x0, y0 = crop_roi(flow, roi)

    if roi_flow is None:
        return None

    h, w = roi_flow.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = roi_flow[y, x]

            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 0.03:
                continue

            x_start = int(x0 + x)
            y_start = int(y0 + y)
            x_end = int(x_start + scale * dx)
            y_end = int(y_start + scale * dy)

            cv2.arrowedLine(
                vis,
                (x_start, y_start),
                (x_end, y_end),
                color,
                1,
                cv2.LINE_AA,
                tipLength=0.3,
            )

    return roi_flow


def check_cuda():
    if not hasattr(cv2, "cuda"):
        raise RuntimeError("OpenCV wurde ohne CUDA-Modul gebaut.")

    if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
        raise RuntimeError("Keine CUDA-fähige GPU gefunden.")

    if not hasattr(cv2.cuda, "FarnebackOpticalFlow_create"):
        raise RuntimeError("cv2.cuda.FarnebackOpticalFlow_create ist nicht verfügbar.")


def main():
    parser = argparse.ArgumentParser(description="Animate CUDA optical flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--scale", type=float, default=8.0)
    args = parser.parse_args()

    check_cuda()

    config = load_config(args.config)

    video_path = config["video_path"]

    rois = {
        "inner_pipe": config.get("inner_pipe_roi"),
        "bed_edge": config.get("bed_edge_roi"),
        "particle_bed": config.get("bed_roi"),
    }

    colors = {
        "inner_pipe": (255, 255, 0),
        "bed_edge": (0, 255, 0),
        "particle_bed": (255, 0, 0),
    }

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

    gpu_prev = cv2.cuda_GpuMat()
    gpu_gray = cv2.cuda_GpuMat()

    gpu_prev.upload(prev_gray)

    cuda_farneback = cv2.cuda.FarnebackOpticalFlow_create(
        3,
        0.5,
        False,
        25,
        3,
        7,
        1.5,
        0,
    )

    frame_idx = start_frame + 1

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, config)
        gray = preprocess(frame)

        gpu_gray.upload(gray)
        gpu_flow = cuda_farneback.calc(gpu_prev, gpu_gray, None)
        flow = gpu_flow.download()

        vis = frame.copy()
        text_y = 30

        for roi_name, roi in rois.items():
            if roi is None:
                continue

            color = colors[roi_name]
            x, y, w, h = map(int, roi)

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis,
                roi_name,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            roi_flow = draw_flow_arrows(
                vis,
                flow,
                roi,
                color,
                step=args.step,
                scale=args.scale,
            )

            if roi_flow is not None:
                dx = roi_flow[:, :, 0]
                dy = roi_flow[:, :, 1]
                mag = np.sqrt(dx ** 2 + dy ** 2)

                txt = (
                    f"{roi_name}: "
                    f"dx={np.mean(dx):.2f} "
                    f"dy={np.mean(dy):.2f} "
                    f"mag={np.mean(mag):.2f}"
                )

                cv2.putText(
                    vis,
                    txt,
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                text_y += 25

        cv2.putText(
            vis,
            f"CUDA Optical Flow | Frame {frame_idx}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

        gpu_prev.upload(gray)
        frame_idx += 1

    cap.release()
    writer.release()

    print("CUDA Animation gespeichert.")
    print(args.out)


if __name__ == "__main__":
    main()