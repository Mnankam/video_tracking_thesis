import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_fft(time, signal, max_freq=50.0):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    valid = np.isfinite(time) & np.isfinite(signal)
    time = time[valid]
    signal = signal[valid]

    if len(time) < 10:
        return np.nan, None, None

    dt = np.mean(np.diff(time))
    if dt <= 0:
        return np.nan, None, None

    coeffs = np.polyfit(time, signal, 1)
    trend = np.polyval(coeffs, time)

    signal = signal - trend
    signal = signal - np.mean(signal)

    freq = np.fft.rfftfreq(len(signal), d=dt)
    spectrum = np.abs(np.fft.rfft(signal))

    valid_freq = (freq > 0) & (freq <= max_freq)

    if not np.any(valid_freq):
        return np.nan, freq, spectrum

    dominant_frequency = freq[valid_freq][np.argmax(spectrum[valid_freq])]

    return float(dominant_frequency), freq, spectrum


def save_fft_plot(freq, spectrum, path, title, max_freq=50.0):
    if freq is None or spectrum is None:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(freq, spectrum)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, max_freq)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyse CUDA optical flow")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-freq", type=float, default=50.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"time_seconds", "roi_name", "mean_dx", "mean_dy", "mean_magnitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    summary = []

    for roi_name in sorted(df["roi_name"].unique()):
        d = df[df["roi_name"] == roi_name].copy()

        d["cum_dx"] = d["mean_dx"].cumsum()
        d["cum_dy"] = d["mean_dy"].cumsum()

        signals = {
            "mean_dx": d["mean_dx"],
            "mean_dy": d["mean_dy"],
            "mean_magnitude": d["mean_magnitude"],
            "cum_dx": d["cum_dx"],
            "cum_dy": d["cum_dy"],
        }

        roi_out_dir = os.path.join(args.out_dir, roi_name)
        os.makedirs(roi_out_dir, exist_ok=True)

        for signal_name, signal in signals.items():
            amp = 0.5 * (signal.max() - signal.min())
            rms = np.sqrt(np.mean(signal ** 2))

            dom_freq, freq, spectrum = compute_fft(
                d["time_seconds"].values,
                signal.values,
                max_freq=args.max_freq,
            )

            save_fft_plot(
                freq,
                spectrum,
                os.path.join(roi_out_dir, f"{signal_name}_fft.png"),
                f"CUDA FFT: {roi_name} - {signal_name}",
                max_freq=args.max_freq,
            )

            summary.append(
                {
                    "method": "cuda_farneback_dense_gpu",
                    "roi_name": roi_name,
                    "signal": signal_name,
                    "amplitude_px": float(amp),
                    "rms_px": float(rms),
                    "dominant_frequency_hz": dom_freq,
                    "mean_value": float(signal.mean()),
                    "std_value": float(signal.std()),
                    "num_samples": int(len(signal)),
                }
            )

    if "processing_time_s" in df.columns:
        summary.append(
            {
                "method": "cuda_farneback_dense_gpu",
                "roi_name": "all",
                "signal": "processing_time_s",
                "amplitude_px": "",
                "rms_px": "",
                "dominant_frequency_hz": "",
                "mean_value": float(df["processing_time_s"].mean()),
                "std_value": float(df["processing_time_s"].std()),
                "num_samples": int(len(df)),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.out_dir, "cuda_motion_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("CUDA Analyse abgeschlossen.")
    print(f"Summary: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()