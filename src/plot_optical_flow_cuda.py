import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(df, y_col, title, ylabel, out_path):
    plt.figure(figsize=(10, 5))

    for roi_name in sorted(df["roi_name"].unique()):
        d = df[df["roi_name"] == roi_name]
        plt.plot(d["time_seconds"], d[y_col], label=roi_name)

    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot CUDA optical flow results")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"time_seconds", "roi_name", "mean_dx", "mean_dy", "mean_magnitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    save_plot(
        df,
        "mean_dx",
        "CUDA Optical Flow: mean dx",
        "mean dx [px/frame]",
        os.path.join(args.out_dir, "cuda_mean_dx.png"),
    )

    save_plot(
        df,
        "mean_dy",
        "CUDA Optical Flow: mean dy",
        "mean dy [px/frame]",
        os.path.join(args.out_dir, "cuda_mean_dy.png"),
    )

    save_plot(
        df,
        "mean_magnitude",
        "CUDA Optical Flow: mean magnitude",
        "mean magnitude [px/frame]",
        os.path.join(args.out_dir, "cuda_mean_magnitude.png"),
    )

    if "processing_time_s" in df.columns:
        save_plot(
            df,
            "processing_time_s",
            "CUDA Optical Flow: processing time",
            "processing time [s/frame]",
            os.path.join(args.out_dir, "cuda_processing_time.png"),
        )

    print(f"CUDA plots gespeichert in: {args.out_dir}")


if __name__ == "__main__":
    main()