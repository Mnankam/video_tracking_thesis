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
    return df


def plot_tracks_per_frame(df: pd.DataFrame, output_dir: str) -> bool:
    if "track_id" not in df.columns or "frame" not in df.columns:
        print("Spalten 'track_id' oder 'frame' fehlen. Tracks-per-frame-Plot wird übersprungen.")
        return False

    df_tracks = df[df["track_id"] != -1]
    if df_tracks.empty:
        print("Keine gültigen Tracks für Tracks-per-frame-Plot gefunden.")
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
        print("Spalten für Trajektorien fehlen. Trajektorien-Plot wird übersprungen.")
        return False

    df_tracks = df[df["track_id"] != -1]

    if df_tracks.empty:
        print("Keine gültigen Tracks für Trajektorien-Plot gefunden.")
        return False

    plt.figure(figsize=(8, 6))

    track_ids = df_tracks["track_id"].dropna().unique()[:max_tracks]

    for tid in track_ids:
        track = df_tracks[df_tracks["track_id"] == tid]
        plt.plot(track["center_x"], track["center_y"], label=f"ID {int(tid)}")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Trajektorien (erste {len(track_ids)} Tracks)")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "trajectories.png")
    plt.savefig(out_path)
    plt.close()
    return True


def plot_bed_edge(df: pd.DataFrame, output_dir: str) -> bool:
    if "bed_edge_y" not in df.columns or "frame" not in df.columns:
        print("Spalte 'bed_edge_y' fehlt. Bettkanten-Plot wird übersprungen.")
        return False

    bed_df = df[["frame", "bed_edge_y"]].copy()
    bed_df["bed_edge_y"] = pd.to_numeric(bed_df["bed_edge_y"], errors="coerce")
    bed_df = bed_df.dropna()

    if bed_df.empty:
        print("Keine gültigen Bettkanten-Daten gefunden.")
        return False

    bed_df = bed_df.groupby("frame", as_index=False)["bed_edge_y"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(bed_df["frame"], bed_df["bed_edge_y"])
    plt.xlabel("Frame")
    plt.ylabel("Bettkante y")
    plt.title("Bettkante über die Zeit")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "bed_edge_y.png")
    plt.savefig(out_path)
    plt.close()
    return True


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

    saved_files = []

    if plot_tracks_per_frame(df, output_dir):
        saved_files.append(os.path.join(output_dir, "tracks_per_frame.png"))

    if plot_trajectories(df, output_dir, max_tracks=10):
        saved_files.append(os.path.join(output_dir, "trajectories.png"))

    if plot_bed_edge(df, output_dir):
        saved_files.append(os.path.join(output_dir, "bed_edge_y.png"))

    if saved_files:
        print("Plots gespeichert:")
        for path in saved_files:
            print(f"- {path}")
    else:
        print("Es wurden keine Plots erzeugt.")


if __name__ == "__main__":
    main()