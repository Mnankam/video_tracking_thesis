import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
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


def crop_roi(arr, roi):
    if roi is None:
        return arr

    x, y, w, h = map(int, roi)
    H, W = arr.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return arr[y:y + h, x:x + w]


def check_cuda():
    if not hasattr(cv2, "cuda"):
        raise RuntimeError("OpenCV wurde ohne CUDA-Modul gebaut.")

    if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
        raise RuntimeError("Keine CUDA-fähige GPU gefunden.")

    if not hasattr(cv2.cuda, "FarnebackOpticalFlow_create"):
        raise RuntimeError("cv2.cuda.FarnebackOpticalFlow_create ist nicht verfügbar.")


def main():
    parser = argparse.ArgumentParser(description="CUDA dense optical flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    check_cuda()

    config = load_config(args.config)

    video_path = config["video_path"]
    output_csv = args.output_csv or config.get("optical_flow_csv")

    if output_csv is None:
        raise ValueError("output csv fehlt.")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_frame = int(config.get("start_frame", 0))
    end_frame = config.get("end_frame")

    rois = {
        "inner_pipe": config.get("inner_pipe_roi"),
        "bed_edge": config.get("bed_edge_roi"),
        "particle_bed": config.get("bed_roi"),
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(int(end_frame), total_frames)

    if start_frame >= end_frame:
        raise ValueError(f"Ungültiger Framebereich: start={start_frame}, end={end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, prev_frame = cap.read()
    if not ok:
        raise RuntimeError("Startframe konnte nicht gelesen werden.")

    prev_frame = resize_frame(prev_frame, config)
    prev_gray = preprocess(prev_frame)

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

    rows = []
    frame_idx = start_frame + 1

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, config)
        gray = preprocess(frame)

        gpu_gray.upload(gray)

        t0 = time.perf_counter()
        gpu_flow = cuda_farneback.calc(gpu_prev, gpu_gray, None)
        flow = gpu_flow.download()
        processing_time_s = time.perf_counter() - t0

        for roi_name, roi in rois.items():
            if roi is None:
                continue

            roi_flow = crop_roi(flow, roi)

            dx = roi_flow[:, :, 0]
            dy = roi_flow[:, :, 1]
            mag = np.sqrt(dx ** 2 + dy ** 2)

            rows.append(
                {
                    "method": "cuda_farneback_dense_gpu",
                    "frame": frame_idx,
                    "time_seconds": frame_idx / fps if fps > 0 else 0.0,
                    "roi_name": roi_name,
                    "mean_dx": float(np.mean(dx)),
                    "mean_dy": float(np.mean(dy)),
                    "median_dx": float(np.median(dx)),
                    "median_dy": float(np.median(dy)),
                    "mean_magnitude": float(np.mean(mag)),
                    "median_magnitude": float(np.median(mag)),
                    "max_magnitude": float(np.max(mag)),
                    "processing_time_s": float(processing_time_s),
                }
            )

        gpu_prev.upload(gray)
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("CUDA Optical Flow hat keine Daten erzeugt.")

    df.to_csv(output_csv, index=False)

    print("CUDA Optical Flow abgeschlossen.")
    print(f"Video: {video_path}")
    print(f"Output: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()