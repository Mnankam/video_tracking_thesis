from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


# =========================================================
# HILFSFUNKTIONEN
# =========================================================
def draw_roi(frame, roi, color, label: str) -> None:
    if roi is None:
        return

    try:
        x, y, w, h = map(int, roi)
    except Exception:
        return

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(20, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def get_bed_edge_column(df: pd.DataFrame) -> Optional[str]:
    if "bed_edge_y_smooth" in df.columns:
        return "bed_edge_y_smooth"
    if "bed_edge_y_raw" in df.columns:
        return "bed_edge_y_raw"
    if "bed_edge_y" in df.columns:
        return "bed_edge_y"
    return None


# =========================================================
# STANDARD PLOTS
# =========================================================
def plot_tracks_per_frame(df: pd.DataFrame, output_dir: str) -> bool:
    if "track_id" not in df.columns or "frame" not in df.columns:
        return False

    df_tracks = df[df["track_id"] != -1]
    if df_tracks.empty:
        return False

    counts = df_tracks.groupby("frame")["track_id"].count()

    plt.figure(figsize=(10, 5))
    plt.plot(counts.index, counts.values)
    plt.xlabel("Frame")
    plt.ylabel("Anzahl Tracks")
    plt.title("Tracks pro Frame")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "tracks_per_frame.png")
    plt.savefig(out_path)
    plt.close()
    return True


def plot_trajectories(df: pd.DataFrame, output_dir: str, max_tracks: int = 10) -> bool:
    required_cols = {"track_id", "center_x", "center_y"}
    if not required_cols.issubset(df.columns):
        return False

    df_tracks = df[df["track_id"] != -1]
    if df_tracks.empty:
        return False

    plt.figure(figsize=(8, 6))

    track_ids = df_tracks["track_id"].dropna().unique()[:max_tracks]

    for tid in track_ids:
        track = df_tracks[df_tracks["track_id"] == tid]
        plt.plot(track["center_x"], track["center_y"], label=f"ID {int(tid)}")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajektorien")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "trajectories.png")
    plt.savefig(out_path)
    plt.close()
    return True


def plot_bed_edge(df: pd.DataFrame, output_dir: str) -> bool:
    bed_col = get_bed_edge_column(df)
    if bed_col is None:
        return False

    bed_df = df[["frame", bed_col]].copy()
    bed_df[bed_col] = pd.to_numeric(bed_df[bed_col], errors="coerce")
    bed_df = bed_df.dropna()

    if bed_df.empty:
        return False

    bed_df = bed_df.groupby("frame", as_index=False)[bed_col].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(bed_df["frame"], bed_df[bed_col])
    plt.xlabel("Frame")
    plt.ylabel("Bettkante y")
    plt.title(f"Bettkante über Zeit ({bed_col})")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "bed_edge_y.png")
    plt.savefig(out_path)
    plt.close()
    return True


def plot_bed_edge_raw_vs_smooth(df: pd.DataFrame, output_dir: str) -> bool:
    if "bed_edge_y_raw" not in df.columns or "bed_edge_y_smooth" not in df.columns:
        return False

    bed_df = df[["frame", "bed_edge_y_raw", "bed_edge_y_smooth"]].copy()
    bed_df["bed_edge_y_raw"] = pd.to_numeric(
        bed_df["bed_edge_y_raw"], errors="coerce"
    )
    bed_df["bed_edge_y_smooth"] = pd.to_numeric(
        bed_df["bed_edge_y_smooth"], errors="coerce"
    )

    bed_df = bed_df.dropna(subset=["bed_edge_y_raw", "bed_edge_y_smooth"])

    if bed_df.empty:
        return False

    bed_df = bed_df.groupby("frame", as_index=False)[
        ["bed_edge_y_raw", "bed_edge_y_smooth"]
    ].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(bed_df["frame"], bed_df["bed_edge_y_raw"], label="raw")
    plt.plot(bed_df["frame"], bed_df["bed_edge_y_smooth"], label="smooth")
    plt.xlabel("Frame")
    plt.ylabel("Bettkante y")
    plt.title("Bettkante: roh vs. geglättet")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "bed_edge_raw_vs_smooth.png")
    plt.savefig(out_path)
    plt.close()
    return True


def plot_inner_pipe_center(df: pd.DataFrame, output_dir: str) -> bool:
    required_cols = {"frame", "inner_pipe_center_x", "inner_pipe_center_y"}
    if not required_cols.issubset(df.columns):
        return False

    ip_df = df[["frame", "inner_pipe_center_x", "inner_pipe_center_y"]].copy()
    ip_df["inner_pipe_center_x"] = pd.to_numeric(
        ip_df["inner_pipe_center_x"], errors="coerce"
    )
    ip_df["inner_pipe_center_y"] = pd.to_numeric(
        ip_df["inner_pipe_center_y"], errors="coerce"
    )
    ip_df = ip_df.dropna()

    if ip_df.empty:
        return False

    ip_df = ip_df.groupby("frame", as_index=False)[
        ["inner_pipe_center_x", "inner_pipe_center_y"]
    ].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(ip_df["frame"], ip_df["inner_pipe_center_x"], label="inner_pipe_center_x")
    plt.plot(ip_df["frame"], ip_df["inner_pipe_center_y"], label="inner_pipe_center_y")
    plt.xlabel("Frame")
    plt.ylabel("Position [px]")
    plt.title("Inneres Rohr: Mittelpunkt über Zeit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "inner_pipe_center.png")
    plt.savefig(out_path)
    plt.close()
    return True


# =========================================================
# COLOR EXPERIMENT PLOTS
# =========================================================
def plot_color_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
) -> bool:
    if metric not in df.columns or "color_mode" not in df.columns:
        print(f"{metric} fehlt -> übersprungen")
        return False

    values = pd.to_numeric(df[metric], errors="coerce")
    valid_df = df.copy()
    valid_df[metric] = values
    valid_df = valid_df.dropna(subset=[metric])

    if valid_df.empty:
        print(f"{metric} hat keine gültigen Werte")
        return False

    plt.figure(figsize=(8, 5))
    plt.bar(valid_df["color_mode"], valid_df[metric])
    plt.title(title)
    plt.xlabel("Color Mode")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return True


def plot_color_experiment(output_dir: str) -> List[str]:
    saved_files: List[str] = []

    experiment_dir = os.path.join(output_dir, "color_experiment")
    input_csv = os.path.join(experiment_dir, "color_mode_comparison.csv")

    if not os.path.exists(input_csv):
        print("Kein Farbexperiment gefunden.")
        return saved_files

    df = pd.read_csv(input_csv)

    plots_dir = os.path.join(experiment_dir, "color_plots")
    os.makedirs(plots_dir, exist_ok=True)

    files = [
        (
            "mean_temporal_iou",
            "Temporal IoU",
            "IoU",
            os.path.join(plots_dir, "temporal_iou.png"),
        ),
        (
            "average_track_length",
            "Track Length",
            "Frames",
            os.path.join(plots_dir, "track_length.png"),
        ),
        (
            "num_large_jumps",
            "Large Jumps",
            "Count",
            os.path.join(plots_dir, "jumps.png"),
        ),
        (
            "std_bed_edge_y",
            "Bed Edge Std",
            "Pixel",
            os.path.join(plots_dir, "bed_edge_std.png"),
        ),
        (
            "avg_pipeline_fps",
            "FPS",
            "Frames/s",
            os.path.join(plots_dir, "fps.png"),
        ),
    ]

    for metric, title, ylabel, path in files:
        if plot_color_metric(df, metric, title, ylabel, path):
            saved_files.append(path)

    if saved_files:
        print("Color experiment plots gespeichert in:", plots_dir)

    return saved_files


# =========================================================
# VIDEO OVERLAY
# =========================================================
def build_frame_lookup(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    return {int(k): v for k, v in df.groupby("frame")}


def draw_bed_edge_overlay(frame, rows: pd.DataFrame, width: int) -> None:
    bed_col = get_bed_edge_column(rows)
    if bed_col is None:
        return

    vals = pd.to_numeric(rows[bed_col], errors="coerce").dropna()
    if vals.empty:
        return

    y = int(vals.mean())

    cv2.line(frame, (0, y), (width - 1, y), (255, 0, 0), 2)
    cv2.putText(
        frame,
        f"{bed_col}={y}",
        (10, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    if "bed_edge_y_raw" in rows.columns and "bed_edge_y_smooth" in rows.columns:
        raw_vals = pd.to_numeric(rows["bed_edge_y_raw"], errors="coerce").dropna()
        if not raw_vals.empty:
            y_raw = int(raw_vals.mean())
            cv2.line(frame, (0, y_raw), (width - 1, y_raw), (0, 0, 255), 1)
            cv2.putText(
                frame,
                f"raw={y_raw}",
                (10, max(45, y_raw + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )


def draw_inner_pipe_overlay(frame, rows: pd.DataFrame) -> None:
    required = {"inner_pipe_x", "inner_pipe_y", "inner_pipe_w", "inner_pipe_h"}
    if not required.issubset(rows.columns):
        return

    ip = rows.iloc[0]

    try:
        ip_x = int(float(ip["inner_pipe_x"]))
        ip_y = int(float(ip["inner_pipe_y"]))
        ip_w = int(float(ip["inner_pipe_w"]))
        ip_h = int(float(ip["inner_pipe_h"]))
    except Exception:
        return

    cv2.rectangle(
        frame,
        (ip_x, ip_y),
        (ip_x + ip_w, ip_y + ip_h),
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "inner_pipe",
        (ip_x, max(20, ip_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if {"inner_pipe_center_x", "inner_pipe_center_y"}.issubset(rows.columns):
        try:
            ip_cx = int(float(ip["inner_pipe_center_x"]))
            ip_cy = int(float(ip["inner_pipe_center_y"]))
            cv2.circle(frame, (ip_cx, ip_cy), 5, (255, 255, 0), -1)
        except Exception:
            pass


def render_overlay_video(
    video_path: str,
    df: pd.DataFrame,
    output_video_path: str,
    config: dict,
    max_traj_length: int = 100,
) -> bool:
    required_cols = {"frame", "track_id", "center_x", "center_y"}
    if not required_cols.issubset(df.columns):
        print("Fehlende Spalten für Overlay.")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return False

    start_frame = int(config.get("start_frame", 0))
    end_frame = config.get("end_frame", None)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    end_frame = min(int(end_frame), total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if "resize_width" in config and "resize_height" in config:
        width = int(config["resize_width"])
        height = int(config["resize_height"])
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Overlay-Video konnte nicht geschrieben werden.")
        cap.release()
        return False

    frame_lookup = build_frame_lookup(df)
    trajectories: Dict[int, List[Tuple[int, int]]] = {}

    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if "resize_width" in config and "resize_height" in config:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # ROIs immer zeichnen
        draw_roi(frame, config.get("bed_roi"), (0, 255, 255), "bed_roi")
        draw_roi(frame, config.get("bed_edge_roi"), (255, 0, 0), "bed_edge_roi")
        draw_roi(frame, config.get("inner_pipe_roi"), (255, 255, 0), "inner_pipe_roi")
        draw_roi(frame, config.get("pipe_roi"), (0, 255, 0), "pipe_roi")
        draw_roi(frame, config.get("optical_box_roi"), (255, 0, 255), "optical_box_roi")

        if frame_idx in frame_lookup:
            rows = frame_lookup[frame_idx]

            # Bettkante
            draw_bed_edge_overlay(frame, rows, width)

            # Inneres Rohr
            draw_inner_pipe_overlay(frame, rows)

            # Tracks
            for _, row in rows.iterrows():
                try:
                    track_id = int(row["track_id"])
                except Exception:
                    continue

                if track_id == -1:
                    continue

                try:
                    cx = int(float(row["center_x"]))
                    cy = int(float(row["center_y"]))
                except Exception:
                    continue

                x = row.get("x", None)
                y = row.get("y", None)
                w = row.get("w", None)
                h = row.get("h", None)

                if track_id not in trajectories:
                    trajectories[track_id] = []

                trajectories[track_id].append((cx, cy))
                trajectories[track_id] = trajectories[track_id][-max_traj_length:]

                if pd.notna(x) and pd.notna(y) and pd.notna(w) and pd.notna(h):
                    try:
                        x = int(float(x))
                        y = int(float(y))
                        w = int(float(w))
                        h = int(float(h))
                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2,
                        )
                    except Exception:
                        pass

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                cv2.putText(
                    frame,
                    f"id={track_id}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                traj = trajectories[track_id]
                for i in range(1, len(traj)):
                    cv2.line(frame, traj[i - 1], traj[i], (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"frame={frame_idx}",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pipeline results")
    parser.add_argument("--config", required=True, help="Pfad zur config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    results_csv = config["output_csv"]
    output_dir = os.path.dirname(results_csv) or "outputs"
    os.makedirs(output_dir, exist_ok=True)

    df = load_results(results_csv)

    if df.empty:
        print("Keine Daten in results.csv gefunden.")
        return

    results: List[str] = []

    if plot_tracks_per_frame(df, output_dir):
        results.append(os.path.join(output_dir, "tracks_per_frame.png"))

    if plot_trajectories(df, output_dir):
        results.append(os.path.join(output_dir, "trajectories.png"))

    if plot_bed_edge(df, output_dir):
        results.append(os.path.join(output_dir, "bed_edge_y.png"))

    if plot_bed_edge_raw_vs_smooth(df, output_dir):
        results.append(os.path.join(output_dir, "bed_edge_raw_vs_smooth.png"))

    if plot_inner_pipe_center(df, output_dir):
        results.append(os.path.join(output_dir, "inner_pipe_center.png"))

    video_out = os.path.join(output_dir, "overlay_video.mp4")
    if render_overlay_video(config["video_path"], df, video_out, config):
        results.append(video_out)

    results.extend(plot_color_experiment(output_dir))

    if results:
        print("Erzeugt:")
        for r in results:
            print("-", r)
    else:
        print("Keine Outputs erzeugt.")


if __name__ == "__main__":
    main()