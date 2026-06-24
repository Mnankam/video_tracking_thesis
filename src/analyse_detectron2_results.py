import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    avg_time = df["inference_time_s"].mean() \
        if "inference_time_s" in df.columns else None

    fps = 1.0 / avg_time \
        if avg_time is not None and avg_time > 0 else None
    summary = {
        "num_detections": len(df),

        "num_frames_with_detections":
            df["frame"].nunique(),

        "mean_score":
            df["score"].mean()
            if "score" in df.columns
            else None,

        "mean_area":
            df["area"].mean()
            if "area" in df.columns
            else None,

        "mean_inference_time_s":
            df["inference_time_s"].mean()
            if "inference_time_s" in df.columns
            else None,

        "fps_inference":
            1.0 / df["inference_time_s"].mean()
            if "inference_time_s" in df.columns
            and df["inference_time_s"].mean() > 0
            else None,

        "avg_time":
            avg_time,

        "fps":
            fps,

        "total_frames":
            len(df),
    }

    summary_df = pd.DataFrame([summary])

    summary_path = os.path.join(
        args.out_dir,
        "detectron2_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print(summary_df)
    print(f"Summary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()