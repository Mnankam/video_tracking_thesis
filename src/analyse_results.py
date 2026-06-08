import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, path):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fft_plot(time, signal, title, path, max_freq=50):
    signal = np.asarray(signal)
    time = np.asarray(time)

    valid = np.isfinite(signal) & np.isfinite(time)
    signal = signal[valid]
    time = time[valid]

    if len(signal) < 10:
        return None

    signal = signal - np.mean(signal)
    dt = np.mean(np.diff(time))

    freq = np.fft.rfftfreq(len(signal), d=dt)
    spectrum = np.abs(np.fft.rfft(signal))

    dominant_idx = np.argmax(spectrum[1:]) + 1
    dominant_freq = freq[dominant_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(freq, spectrum)
    plt.xlim(0, max_freq)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return dominant_freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--optical-flow-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=200.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = []

    # =====================================================
    # 1. Segmentierung: inner pipe / bed edge / particle bed
    # =====================================================
    results = pd.read_csv(args.results_csv)

    if "time_seconds" not in results.columns:
        results["time_seconds"] = results["frame"] / args.fps

    segmentation_signals = [
        "inner_pipe_center_y",
        "inner_pipe_center_x",
        "bed_edge_y",
        "bed_edge_y_smooth",
        "particle_bed_center_y",
        "particle_bed_center_x",
        "bed_center_y",
        "bed_center_x",
    ]

    for col in segmentation_signals:
        if col in results.columns:
            y = results[col]

            save_plot(
                results["time_seconds"],
                y,
                "Time [s]",
                f"{col} [px]",
                f"Segmentation: {col}",
                os.path.join(args.out_dir, f"segmentation_{col}.png"),
            )

            rel = y - y.dropna().iloc[0] if y.notna().any() else y

            save_plot(
                results["time_seconds"],
                rel,
                "Time [s]",
                f"relative {col} [px]",
                f"Relative segmentation displacement: {col}",
                os.path.join(args.out_dir, f"segmentation_{col}_relative.png"),
            )

            dom_freq = fft_plot(
                results["time_seconds"],
                rel,
                f"FFT: {col}",
                os.path.join(args.out_dir, f"segmentation_{col}_fft.png"),
            )

            summary.append({
                "source": "segmentation",
                "signal": col,
                "amplitude_px": 0.5 * (rel.max() - rel.min()),
                "rms_px": np.sqrt(np.nanmean(rel ** 2)),
                "dominant_frequency_hz": dom_freq,
            })

    # =====================================================
    # 2. Optical Flow: x, y, dx, dy, displacement magnitude
    # =====================================================
    flow = pd.read_csv(args.optical_flow_csv)

    if "tracking_status" in flow.columns:
        flow = flow[flow["tracking_status"] == 1].copy()

    if "time_seconds" not in flow.columns:
        flow["time_seconds"] = flow["frame"] / args.fps

    flow["displacement_magnitude"] = np.sqrt(flow["dx"] ** 2 + flow["dy"] ** 2)

    for pid in sorted(flow["point_id"].unique()):
        d = flow[flow["point_id"] == pid].copy()

        d["x_rel"] = d["x"] - d["x"].iloc[0]
        d["y_rel"] = d["y"] - d["y"].iloc[0]

        for col in ["x", "y", "dx", "dy", "displacement_magnitude", "x_rel", "y_rel"]:
            save_plot(
                d["time_seconds"],
                d[col],
                "Time [s]",
                f"{col} [px]",
                f"Optical Flow point {pid}: {col}",
                os.path.join(args.out_dir, f"flow_point_{pid}_{col}.png"),
            )

        dom_freq = fft_plot(
            d["time_seconds"],
            d["y_rel"],
            f"FFT Optical Flow point {pid}: y displacement",
            os.path.join(args.out_dir, f"flow_point_{pid}_y_fft.png"),
        )

        summary.append({
            "source": "optical_flow",
            "signal": f"point_{pid}_y_rel",
            "amplitude_px": 0.5 * (d["y_rel"].max() - d["y_rel"].min()),
            "rms_px": np.sqrt(np.mean(d["y_rel"] ** 2)),
            "dominant_frequency_hz": dom_freq,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        os.path.join(args.out_dir, "motion_analysis_summary.csv"),
        index=False,
    )

    print("Analyse abgeschlossen.")
    print(summary_df)


if __name__ == "__main__":
    main()