from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

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
# PLOTS
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
    if "bed_edge_y" not in df.columns:
        return False

    bed_df = df[["frame", "bed_edge_y"]].copy()
    bed_df["bed_edge_y"] = pd.to_numeric(bed_df["bed_edge_y"], errors="coerce")
    bed_df = bed_df.dropna()

    if bed_df.empty:
        return False

    bed_df = bed_df.groupby("frame", as_index=False)["bed_edge_y"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(bed_df["frame"], bed_df["bed_edge_y"])
    plt.xlabel("Frame")
    plt.ylabel("Bettkante y")
    plt.title("Bettkante über Zeit")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "bed_edge_y.png")
    plt.savefig(out_path)
    plt.close()
    return True


# =========================================================
# VIDEO OVERLAY
# =========================================================
def build_frame_lookup(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    return {int(k): v for k, v in df.groupby("frame")}


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

    # Resize berücksichtigen
    if "resize_width" in config and "resize_height" in config:
        width = config["resize_width"]
        height = config["resize_height"]
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_lookup = build_frame_lookup(df)
    trajectories: Dict[int, List[Tuple[int, int]]] = {}

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # >>> WICHTIG: Resize wie Pipeline
        if "resize_width" in config and "resize_height" in config:
            frame = cv2.resize(frame, (width, height))

        if frame_idx in frame_lookup:
            rows = frame_lookup[frame_idx]

            # Bettkante
            if "bed_edge_y" in rows.columns:
                vals = pd.to_numeric(rows["bed_edge_y"], errors="coerce").dropna()
                if not vals.empty:
                    y = int(vals.mean())
                    cv2.line(frame, (0, y), (width - 1, y), (255, 0, 0), 2)

            for _, row in rows.iterrows():
                if int(row["track_id"]) == -1:
                    continue

                try:
                    cx = int(float(row["center_x"]))
                    cy = int(float(row["center_y"]))
                except:
                    continue

                track_id = int(row["track_id"])

                # Trajektorie speichern
                if track_id not in trajectories:
                    trajectories[track_id] = []

                trajectories[track_id].append((cx, cy))
                trajectories[track_id] = trajectories[track_id][-max_traj_length:]

                # Punkt
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # ID
                cv2.putText(
                    frame,
                    f"id={track_id}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Trajektorie zeichnen
                traj = trajectories[track_id]
                for i in range(1, len(traj)):
                    cv2.line(frame, traj[i - 1], traj[i], (0, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    df = load_results(config["output_csv"])
    output_dir = os.path.dirname(config["output_csv"]) or "outputs"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    if plot_tracks_per_frame(df, output_dir):
        results.append("tracks_per_frame.png")

    if plot_trajectories(df, output_dir):
        results.append("trajectories.png")

    if plot_bed_edge(df, output_dir):
        results.append("bed_edge_y.png")

    video_out = os.path.join(output_dir, "overlay_video.mp4")
    if render_overlay_video(config["video_path"], df, video_out, config):
        results.append("overlay_video.mp4")

    if results:
        print("Erzeugt:")
        for r in results:
            print("-", r)
    else:
        print("Keine Outputs erzeugt.")


if __name__ == "__main__":
    main()