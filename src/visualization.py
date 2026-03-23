from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "track_id" in df.columns:
        df = df[df["track_id"] != -1]

    return df


def plot_tracks_per_frame(df: pd.DataFrame, output_dir: str) -> None:
    counts = df.groupby("frame")["track_id"].count()

    plt.figure(figsize=(10, 5))
    plt.plot(counts.index, counts.values)
    plt.xlabel("Frame")
    plt.ylabel("Anzahl Tracks")
    plt.title("Tracks pro Frame")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tracks_per_frame.png"))
    plt.close()


def plot_trajectories(df: pd.DataFrame, output_dir: str, max_tracks: int = 10) -> None:
    plt.figure(figsize=(8, 6))

    track_ids = df["track_id"].dropna().unique()[:max_tracks]

    for tid in track_ids:
        track = df[df["track_id"] == tid]
        plt.plot(track["center_x"], track["center_y"], label=f"ID {int(tid)}")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Trajektorien (erste {len(track_ids)} Tracks)")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectories.png"))
    plt.close()


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
        print("Keine gültigen Tracking-Daten in results.csv gefunden.")
        return

    plot_tracks_per_frame(df, output_dir)
    plot_trajectories(df, output_dir, max_tracks=10)

    print("Plots gespeichert:")
    print(f"- {os.path.join(output_dir, 'tracks_per_frame.png')}")
    print(f"- {os.path.join(output_dir, 'trajectories.png')}")


if __name__ == "__main__":
    main()