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
        return cv2.resize(frame, (int(rw), int(rh)), interpolation=cv2.INTER_AREA)

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
        cv2.putText(
            vis,
            name,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


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
    parser = argparse.ArgumentParser(description="Detectron2 inference on flow loop videos")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    video_path = config["video_path"]

    output_csv = (
        args.output_csv
        or config.get("detectron2_output_csv")
        or config.get("output_csv")
    )

    debug_dir = (
        args.debug_dir
        or config.get("detectron2_debug_dir")
        or config.get("debug_dir")
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    start_frame = int(config.get("start_frame", 0))
    end_frame = config.get("end_frame", None)

    predictor = build_predictor(config)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(int(end_frame), total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_frame(frame, config)

        t0 = time.perf_counter()
        outputs = predictor(frame)
        inference_time_s = time.perf_counter() - t0

        instances = outputs["instances"].to("cpu")

        vis = frame.copy()
        draw_roi_boxes(vis, config)

        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy() if instances.has("scores") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)

                w = int(x2 - x1)
                h = int(y2 - y1)
                cx = float(x1 + w / 2.0)
                cy = float(y1 + h / 2.0)

                score = float(scores[i]) if len(scores) > i else None
                class_id = int(classes[i]) if len(classes) > i else None

                rows.append(
                    {
                        "method": "detectron2_mask_rcnn",
                        "frame": frame_idx,
                        "time_seconds": frame_idx / fps if fps > 0 else 0.0,
                        "class_id": class_id,
                        "score": score,
                        "x": int(x1),
                        "y": int(y1),
                        "w": w,
                        "h": h,
                        "center_x": cx,
                        "center_y": cy,
                        "area": float(w * h),
                        "inference_time_s": float(inference_time_s),
                    }
                )

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"id={class_id} score={score:.2f}" if score is not None else f"id={class_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        if frame_idx % 10 == 0:
            cv2.imwrite(
                os.path.join(debug_dir, f"frame_{frame_idx:06d}.png"),
                vis,
            )

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)

    if df.empty:
        print("Warnung: Detectron2 hat keine Objekte erkannt.")
    else:
        df.to_csv(output_csv, index=False)

    print("Detectron2 inference abgeschlossen.")
    print(f"Video: {video_path}")
    print(f"Output: {output_csv}")
    print(f"Debug: {debug_dir}")
    print(f"Frames: {start_frame} bis {frame_idx}")


if __name__ == "__main__":
    main()