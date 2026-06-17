import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    idx = np.argmax(spectrum[valid_freq])
    dominant_frequency = freq[valid_freq][idx]

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
    parser = argparse.ArgumentParser(description="Analyse Farneback dense optical flow")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-freq", type=float, default=50.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"time_seconds", "mean_dx", "mean_dy", "mean_magnitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}")

    df["cum_dx"] = df["mean_dx"].cumsum()
    df["cum_dy"] = df["mean_dy"].cumsum()

    signals = {
        "mean_dx": df["mean_dx"],
        "mean_dy": df["mean_dy"],
        "mean_magnitude": df["mean_magnitude"],
        "cum_dx": df["cum_dx"],
        "cum_dy": df["cum_dy"],
    }

    summary = []

    for name, signal in signals.items():
        amp = 0.5 * (signal.max() - signal.min())
        rms = np.sqrt(np.mean(signal ** 2))

        dom_freq, freq, spectrum = compute_fft(
            df["time_seconds"].values,
            signal.values,
            max_freq=args.max_freq,
        )

        save_fft_plot(
            freq,
            spectrum,
            os.path.join(args.out_dir, f"{name}_fft.png"),
            f"Farneback FFT: {name}",
            max_freq=args.max_freq,
        )

        summary.append(
            {
                "method": "farneback_dense_cpu",
                "signal": name,
                "amplitude_px": amp,
                "rms_px": rms,
                "dominant_frequency_hz": dom_freq,
                "mean_value": signal.mean(),
                "std_value": signal.std(),
                "num_samples": len(signal),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.out_dir, "farneback_motion_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("Farneback Analyse abgeschlossen.")
    print(f"Summary: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()