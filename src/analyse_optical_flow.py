import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    if "tracking_status" in df.columns:
        df = df[df["tracking_status"] == 1].copy()

    summary = []

    for pid in sorted(df["point_id"].unique()):
        d = df[df["point_id"] == pid].copy()

        d["x_rel"] = d["x"] - d["x"].iloc[0]
        d["y_rel"] = d["y"] - d["y"].iloc[0]

        amp_x = 0.5 * (d["x_rel"].max() - d["x_rel"].min())
        amp_y = 0.5 * (d["y_rel"].max() - d["y_rel"].min())

        rms_x = np.sqrt(np.mean(d["x_rel"] ** 2))
        rms_y = np.sqrt(np.mean(d["y_rel"] ** 2))

        summary.append({
            "point_id": pid,
            "amp_x_px": amp_x,
            "amp_y_px": amp_y,
            "rms_x_px": rms_x,
            "rms_y_px": rms_y,
            "mean_x_px": d["x"].mean(),
            "mean_y_px": d["y"].mean(),
        })

        plt.figure(figsize=(10, 5))
        plt.plot(d["time_seconds"], d["y_rel"])
        plt.xlabel("Time [s]")
        plt.ylabel("Relative y displacement [px]")
        plt.title(f"Point {pid}: vertical displacement")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"point_{pid}_y_displacement.png"), dpi=150)
        plt.close()

        # FFT
        t = d["time_seconds"].values
        y = d["y_rel"].values - d["y_rel"].mean()

        if len(t) > 10:
            dt = np.mean(np.diff(t))
            freq = np.fft.rfftfreq(len(y), d=dt)
            spectrum = np.abs(np.fft.rfft(y))

            plt.figure(figsize=(10, 5))
            plt.plot(freq, spectrum)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude")
            plt.title(f"Point {pid}: frequency spectrum")
            plt.grid(True)
            plt.xlim(0, 50)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"point_{pid}_fft.png"), dpi=150)
            plt.close()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(args.out_dir, "optical_flow_motion_summary.csv"), index=False)

    print(summary_df)


if __name__ == "__main__":
    main()