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
        return cv2.resize(
            frame,
            (int(rw), int(rh)),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rauschunterdrückung
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kontrast stabilisieren
    gray = cv2.equalizeHist(gray)

    return gray


def crop_roi(arr, roi):
    if roi is None:
        return None

    x, y, w, h = map(int, roi)

    H, W = arr.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return arr[y:y + h, x:x + w]


def analyse_roi(flow, roi_name, roi, frame_idx, fps, compute_time_s):
    roi_flow = crop_roi(flow, roi)

    if roi_flow is None:
        return None

    dx = roi_flow[:, :, 0]
    dy = roi_flow[:, :, 1]

    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    return {
        "method": "farneback_dense_cpu",
        "frame": frame_idx,
        "time_seconds": frame_idx / fps if fps > 0 else 0.0,
        "roi_name": roi_name,

        "mean_dx": float(np.mean(dx)),
        "mean_dy": float(np.mean(dy)),

        "median_dx": float(np.median(dx)),
        "median_dy": float(np.median(dy)),

        "mean_magnitude": float(np.mean(magnitude)),
        "median_magnitude": float(np.median(magnitude)),
        "max_magnitude": float(np.max(magnitude)),

        "compute_time_s": float(compute_time_s),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Farneback Dense Optical Flow"
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default=None)

    args = parser.parse_args()

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
        raise RuntimeError(
            f"Video konnte nicht geöffnet werden: {video_path}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(int(end_frame), total_frames)

    if start_frame >= end_frame:
        raise ValueError(
            f"Ungültiger Framebereich: start={start_frame}, end={end_frame}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, prev_frame = cap.read()

    if not ok:
        raise RuntimeError("Startframe konnte nicht gelesen werden.")

    prev_frame = resize_frame(prev_frame, config)
    prev_gray = preprocess(prev_frame)

    rows = []
    compute_times = []
    processed_frames = 0

    total_start_time = time.perf_counter()

    frame_idx = start_frame + 1

    while frame_idx < end_frame:

        ok, frame = cap.read()

        if not ok:
            break

        frame = resize_frame(frame, config)
        gray = preprocess(frame)

        compute_start_time = time.perf_counter()

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,      # pyr_scale
            3,        # levels
            25,       # winsize
            3,        # iterations
            7,        # poly_n
            1.5,      # poly_sigma
            0,
        )

        compute_time_s = time.perf_counter() - compute_start_time
        compute_times.append(compute_time_s)

        for roi_name, roi in rois.items():

            result = analyse_roi(
                flow,
                roi_name,
                roi,
                frame_idx,
                fps,
                compute_time_s,
            )

            if result is not None:
                rows.append(result)

        prev_gray = gray.copy()

        processed_frames += 1
        frame_idx += 1

    total_runtime_s = time.perf_counter() - total_start_time

    cap.release()

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Farneback hat keine Daten erzeugt.")

    df.to_csv(output_csv, index=False)

    avg_compute_time_s = float(np.mean(compute_times)) if compute_times else 0.0
    avg_total_frame_time_s = total_runtime_s / processed_frames if processed_frames > 0 else 0.0
    effective_fps = processed_frames / total_runtime_s if total_runtime_s > 0 else 0.0

    benchmark_df = pd.DataFrame(
        [
            {
                "method": "farneback_dense_cpu",
                "video_path": video_path,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "processed_frames": processed_frames,
                "video_fps": fps,
                "total_runtime_s": total_runtime_s,
                "avg_compute_time_s": avg_compute_time_s,
                "avg_total_frame_time_s": avg_total_frame_time_s,
                "effective_fps": effective_fps,
                "num_rois": int(sum(roi is not None for roi in rois.values())),
            }
        ]
    )

    benchmark_csv = output_csv.replace(".csv", "_benchmark.csv")
    benchmark_df.to_csv(benchmark_csv, index=False)

    print("Farneback abgeschlossen.")
    print(f"Video: {video_path}")
    print(f"Frames: {start_frame} bis {end_frame}")
    print(f"Output: {output_csv}")
    print(f"Benchmark: {benchmark_csv}")
    print(f"Processed frames: {processed_frames}")
    print(f"Total runtime [s]: {total_runtime_s:.6f}")
    print(f"Avg compute time/frame [s]: {avg_compute_time_s:.6f}")
    print(f"Effective FPS: {effective_fps:.3f}")
    print(df.head())


if __name__ == "__main__":
    main()