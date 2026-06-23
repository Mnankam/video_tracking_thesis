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
    rw = config.get("resize_width", None)
    rh = config.get("resize_height", None)

    if rw is not None and rh is not None:
        return cv2.resize(frame, (int(rw), int(rh)), interpolation=cv2.INTER_AREA)

    return frame


def preprocess_gray(frame, config):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if config.get("optical_flow_use_blur", True):
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if config.get("optical_flow_use_equalize", True):
        gray = cv2.equalizeHist(gray)

    return gray


def snap_points_to_features(gray, points, search_radius=35):
    snapped = []
    h, w = gray.shape[:2]

    for p in points:
        x, y = p.ravel()
        x = int(x)
        y = int(y)

        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(w, x + search_radius)
        y2 = min(h, y + search_radius)

        patch = gray[y1:y2, x1:x2]

        corners = cv2.goodFeaturesToTrack(
            patch,
            maxCorners=1,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        if corners is not None:
            cx, cy = corners[0, 0]
            snapped.append([x1 + cx, y1 + cy])
        else:
            snapped.append([x, y])

    return np.array(snapped, dtype=np.float32).reshape(-1, 1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default=None)

    args = parser.parse_args()

    config = load_config(args.config)

    video_path = config["video_path"]
    output_csv = args.output_csv or config.get("optical_flow_csv")

    points = np.array(config["optical_flow_points"], dtype=np.float32)
    points = points.reshape(-1, 1, 2)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS))

    start_frame = int(config.get("start_frame", 0))
    end_frame = config.get("end_frame", total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    frame = resize_frame(frame, config)

    prev_gray = preprocess_gray(frame, config)
    prev_points = snap_points_to_features(prev_gray, points)

    lk_params = dict(
        winSize=(31,31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,20,0.01)
    )

    rows = []
    processed_frames = 0

    total_start = time.perf_counter()

    frame_idx = start_frame + 1

    while frame_idx < end_frame:

        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, config)
        gray = preprocess_gray(frame, config)

        frame_start = time.perf_counter()

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            **lk_params
        )

        frame_runtime = time.perf_counter() - frame_start

        for i,(p0,p1) in enumerate(zip(prev_points,next_points)):

            x0,y0 = p0.ravel()
            x1,y1 = p1.ravel()

            rows.append({
                "method":"lucas_kanade_cpu",
                "frame":frame_idx,
                "point_id":i,
                "x":float(x1),
                "y":float(y1),
                "dx":float(x1-x0),
                "dy":float(y1-y0),
                "compute_time_s":frame_runtime
            })

        prev_gray = gray.copy()
        prev_points = next_points.copy()

        processed_frames += 1
        frame_idx += 1

    total_runtime = time.perf_counter() - total_start

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv,index=False)

    summary = pd.DataFrame([{
        "method":"lucas_kanade_cpu",
        "processed_frames":processed_frames,
        "total_runtime_s":total_runtime,
        "avg_frame_time_s":total_runtime/processed_frames,
        "effective_fps":processed_frames/total_runtime
    }])

    benchmark = output_csv.replace(".csv","_benchmark.csv")
    summary.to_csv(benchmark,index=False)

    print("Lucas-Kanade abgeschlossen")
    print(summary)