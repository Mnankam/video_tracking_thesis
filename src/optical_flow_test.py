import argparse
import os

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


def save_initial_debug_image(frame, points, output_csv):
    debug_path = output_csv.replace(".csv", "_initial_points.png")

    debug = frame.copy()

    for i, p in enumerate(points):
        x, y = p.ravel().astype(int)
        cv2.circle(debug, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"id={i}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(debug_path, debug)
    print(f"Initial point debug gespeichert: {debug_path}")


def main():
    parser = argparse.ArgumentParser(description="Optical Flow Tracking")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if not config.get("enable_optical_flow", True):
        print("Optical Flow ist in der Config deaktiviert.")
        return

    video_path = config["video_path"]
    output_csv = args.output_csv or config.get("optical_flow_csv")

    if output_csv is None:
        raise ValueError("optical_flow_csv fehlt.")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    start_frame = int(config.get("start_frame", 0))
    end_frame = config.get("end_frame", None)

    points = np.array(config.get("optical_flow_points", []), dtype=np.float32)

    if len(points) == 0:
        raise ValueError("Keine optical_flow_points in config.yaml gefunden.")

    points = points.reshape(-1, 1, 2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))

    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(int(end_frame), total_frames)

    if start_frame >= end_frame:
        raise ValueError(f"Ungültiger Framebereich: start={start_frame}, end={end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Erster Frame konnte nicht gelesen werden.")

    first_frame = resize_frame(first_frame, config)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    prev_points = snap_points_to_features(
        prev_gray,
        points,
        search_radius=int(config.get("optical_flow_snap_radius", 35)),
    )

    save_initial_debug_image(first_frame, prev_points, output_csv)

    rows = []

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            0.01,
        ),
    )

    max_jump_px = float(config.get("optical_flow_max_jump_px", 8.0))

    frame_idx = start_frame + 1

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, config)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            **lk_params,
        )

        if next_points is None or status is None:
            print(f"Optical Flow abgebrochen bei Frame {frame_idx}.")
            break

        valid_points = prev_points.copy()

        for i, (p0, p1, st) in enumerate(zip(prev_points, next_points, status)):
            x0, y0 = p0.ravel()
            x1, y1 = p1.ravel()

            dx = float(x1 - x0)
            dy = float(y1 - y0)
            jump = float(np.sqrt(dx * dx + dy * dy))

            tracking_ok = int(st[0] == 1 and jump <= max_jump_px)

            if tracking_ok:
                valid_points[i] = p1
            else:
                x1, y1 = x0, y0
                dx, dy = 0.0, 0.0

            rows.append(
                {
                    "frame": frame_idx,
                    "time_seconds": frame_idx / video_fps if video_fps > 0 else 0.0,
                    "point_id": i,
                    "x": float(x1),
                    "y": float(y1),
                    "dx": dx,
                    "dy": dy,
                    "jump_px": jump,
                    "tracking_status": tracking_ok,
                }
            )

        prev_gray = gray.copy()
        prev_points = valid_points.copy()
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Optical Flow hat keine Daten erzeugt.")

    df.to_csv(output_csv, index=False)

    print("Optical Flow abgeschlossen.")
    print(f"Video: {video_path}")
    print(f"Frames: {start_frame} bis {end_frame}")
    print(f"Output: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()