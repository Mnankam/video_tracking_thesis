import argparse
import os
import time

import cv2
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


def draw_roi_boxes(vis, config):
    rois = {
        "inner_pipe_roi": config.get("inner_pipe_roi"),
        "bed_roi": config.get("bed_roi"),
        "bed_edge_roi": config.get("bed_edge_roi"),
    }

    colors = {
        "inner_pipe_roi": (255, 255, 0),
        "bed_roi": (0, 165, 255),
        "bed_edge_roi": (255, 0, 0),
    }

    for name, roi in rois.items():
        if roi is None:
            continue

        x, y, w, h = map(int, roi)
        color = colors[name]

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)


def build_predictor(config):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()

    config_file = config.get(
        "detectron2_config_file",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    )

    weights_file = config.get("detectron2_weights_file", "COCO")
    score_threshold = float(config.get("detectron2_score_threshold", 0.5))

    cfg.merge_from_file(model_zoo.get_config_file(config_file))

    if weights_file == "COCO":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    else:
        cfg.MODEL.WEIGHTS = weights_file

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cuda"

    return DefaultPredictor(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--debug-dir", default=None)

    args = parser.parse_args()

    config = load_config(args.config)

    video_path = config["video_path"]

    output_csv = args.output_csv or config.get("detectron2_output_csv")
    debug_dir = args.debug_dir or config.get("detectron2_debug_dir")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    predictor = build_predictor(config)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Video konnte nicht geöffnet werden")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = float(cap.get(cv2.CAP_PROP_FPS))

    start_frame = int(config.get("start_frame", 0))
    end_frame = config.get("end_frame")

    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(int(end_frame), total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows = []
    processed_frames = 0

    total_start = time.perf_counter()

    frame_idx = start_frame

    while frame_idx < end_frame:

        ok, frame = cap.read()

        if not ok:
            break

        frame = resize_frame(frame, config)

        frame_start = time.perf_counter()

        outputs = predictor(frame)

        frame_runtime = time.perf_counter() - frame_start

        instances = outputs["instances"].to("cpu")

        vis = frame.copy()
        draw_roi_boxes(vis, config)

        if instances.has("pred_boxes"):

            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = box.astype(int)

                w = int(x2 - x1)
                h = int(y2 - y1)

                rows.append(
                    {
                        "method": "detectron2_gpu",
                        "frame": frame_idx,
                        "video_time_seconds": frame_idx / fps_video,
                        "class_id": int(classes[i]),
                        "score": float(scores[i]),
                        "x": x1,
                        "y": y1,
                        "w": w,
                        "h": h,
                        "compute_time_s": frame_runtime,
                    }
                )

        if frame_idx % 10 == 0:
            cv2.imwrite(
                os.path.join(
                    debug_dir,
                    f"frame_{frame_idx:06d}.png"
                ),
                vis,
            )

        processed_frames += 1
        frame_idx += 1

    total_runtime = time.perf_counter() - total_start

    cap.release()

    df = pd.DataFrame(rows)

    if not df.empty:
        df.to_csv(output_csv, index=False)

    avg_frame_time = total_runtime / processed_frames
    effective_fps = processed_frames / total_runtime

    summary = pd.DataFrame(
        [
            {
                "method": "detectron2_gpu",
                "processed_frames": processed_frames,
                "total_runtime_s": total_runtime,
                "avg_frame_time_s": avg_frame_time,
                "effective_fps": effective_fps,
            }
        ]
    )

    summary_path = output_csv.replace(".csv", "_benchmark.csv")
    summary.to_csv(summary_path, index=False)

    print("======================================")
    print("Detectron2 Benchmark")
    print("======================================")
    print("Processed frames:", processed_frames)
    print("Total runtime:", total_runtime)
    print("Average frame time:", avg_frame_time)
    print("Effective FPS:", effective_fps)
    print("CSV:", output_csv)
    print("Benchmark:", summary_path)


if __name__ == "__main__":
    main()