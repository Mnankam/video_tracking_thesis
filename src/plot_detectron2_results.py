import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    if df.empty:
        raise RuntimeError("CSV ist leer.")

    plt.figure(figsize=(10, 5))
    plt.plot(df["frame"], df["score"])
    plt.xlabel("Frame")
    plt.ylabel("Detection score")
    plt.title("Detectron2 detection confidence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "detectron2_scores.png"), dpi=150)
    plt.close()

    if "inference_time_s" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["frame"], df["inference_time_s"])
        plt.xlabel("Frame")
        plt.ylabel("Inference time [s]")
        plt.title("Detectron2 inference time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "detectron2_inference_time.png"), dpi=150)
        plt.close()

    print(f"Plots gespeichert: {args.out_dir}")


if __name__ == "__main__":
    main()