from __future__ import annotations

import copy
import csv
import os
from pathlib import Path

from src.pipeline import load_config, VideoPipeline


COLOR_MODES = ["gray", "r", "g", "b", "hsv_v", "hsv_s"]


def read_summary_csv(path: str) -> dict:
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["metric"]] = row["value"]
    return metrics


def main() -> None:
    config_path = "configs/config.yaml"
    base_config = load_config(config_path)

    results_dir = Path("outputs/color_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    for mode in COLOR_MODES:
        print(f"\n=== Running color mode: {mode} ===")

        config = copy.deepcopy(base_config)
        config.segmentation_color_mode = mode
        config.output_csv = str(results_dir / f"results_{mode}.csv")
        config.summary_csv = str(results_dir / f"summary_{mode}.csv")
        config.debug_dir = str(results_dir / f"debug_{mode}")

        pipeline = VideoPipeline(config)
        summary = pipeline.run()

        comparison_rows.append(
            {
                "color_mode": mode,
                "avg_processing_time_s": summary.get("avg_processing_time_s", ""),
                "avg_pipeline_fps": summary.get("avg_pipeline_fps", ""),
                "avg_track_count": summary.get("avg_track_count", ""),
                "average_track_length": summary.get("average_track_length", ""),
                "num_unique_tracks": summary.get("num_unique_tracks", ""),
                "num_large_jumps": summary.get("num_large_jumps", ""),
                "mean_bed_edge_y": summary.get("mean_bed_edge_y", ""),
                "std_bed_edge_y": summary.get("std_bed_edge_y", ""),
                "mean_temporal_iou": summary.get("mean_temporal_iou", ""),
            }
        )

    comparison_csv = results_dir / "color_mode_comparison.csv"
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\nColor comparison finished.")
    print(f"Saved comparison table to: {comparison_csv}")


if __name__ == "__main__":
    main()