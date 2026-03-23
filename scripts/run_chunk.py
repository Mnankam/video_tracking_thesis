from __future__ import annotations

import argparse
import os

from src.pipeline import PipelineConfig, VideoPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline on a frame chunk")
    parser.add_argument("--video", required=True, help="Pfad zum Video")
    parser.add_argument("--out", required=True, help="Pfad zur Output-CSV")
    parser.add_argument("--start", type=int, required=True, help="Start-Frame")
    parser.add_argument("--end", type=int, required=True, help="End-Frame")
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    config = PipelineConfig(
        video_path=args.video,
        output_csv=args.out,
        start_frame=args.start,
        end_frame=args.end,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        enable_deflicker=True,
        enable_tracking=True,
        save_debug_frames=args.debug,
        debug_dir="outputs/debug",
    )

    pipeline = VideoPipeline(config)
    summary = pipeline.run()

    print("Chunk processing finished.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()