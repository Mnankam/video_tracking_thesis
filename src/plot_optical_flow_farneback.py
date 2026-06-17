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

    required = {
        "time_seconds",
        "mean_dx",
        "mean_dy",
        "mean_magnitude",
        "max_magnitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    # -----------------------------------------------------
    # Plot 1: Mean dx
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_seconds"], df["mean_dx"])
    plt.xlabel("Time [s]")
    plt.ylabel("Mean dx [px]")
    plt.title("Farneback Mean Horizontal Motion")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, "farneback_mean_dx.png"),
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # Plot 2: Mean dy
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_seconds"], df["mean_dy"])
    plt.xlabel("Time [s]")
    plt.ylabel("Mean dy [px]")
    plt.title("Farneback Mean Vertical Motion")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, "farneback_mean_dy.png"),
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # Plot 3: Mean Magnitude
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_seconds"], df["mean_magnitude"])
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude [px]")
    plt.title("Farneback Mean Motion Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, "farneback_mean_magnitude.png"),
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # Plot 4: Max Magnitude
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_seconds"], df["max_magnitude"])
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude [px]")
    plt.title("Farneback Max Motion Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, "farneback_max_magnitude.png"),
        dpi=150,
    )
    plt.close()

    print("Farneback plots gespeichert.")
    print(args.out_dir)


if __name__ == "__main__":
    main()