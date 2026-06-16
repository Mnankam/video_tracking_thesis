import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_dominant_frequency(time, signal, max_freq=50.0):
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

    # Linearen Trend entfernen
    coeffs = np.polyfit(time, signal, 1)
    trend = np.polyval(coeffs, time)
    signal_detrended = signal - trend

    # Mittelwert entfernen
    signal_detrended = signal_detrended - np.mean(signal_detrended)

    freq = np.fft.rfftfreq(len(signal_detrended), d=dt)
    spectrum = np.abs(np.fft.rfft(signal_detrended))

    valid_freq = (freq > 0) & (freq <= max_freq)

    if not np.any(valid_freq):
        return np.nan, freq, spectrum

    dominant_idx_local = np.argmax(spectrum[valid_freq])
    dominant_frequency = freq[valid_freq][dominant_idx_local]

    return float(dominant_frequency), freq, spectrum


def save_signal_plot(time, signal, path, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
    parser = argparse.ArgumentParser(description="Analyse Optical Flow motion")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-freq", type=float, default=50.0)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    parser.add_argument("--max-mean-jump", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_all = pd.read_csv(args.csv)

    required_cols = {"frame", "time_seconds", "point_id", "x", "y"}
    missing = required_cols - set(df_all.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}")

    summary = []

    for pid in sorted(df_all["point_id"].unique()):
        d_all = df_all[df_all["point_id"] == pid].copy()

        total_count = len(d_all)

        if "tracking_status" in d_all.columns:
            valid_mask = d_all["tracking_status"] == 1
            success_rate = valid_mask.mean()
            d = d_all[valid_mask].copy()
        else:
            success_rate = 1.0
            d = d_all.copy()

        if d.empty or len(d) < 10:
            summary.append({
                "point_id": pid,
                "valid": False,
                "reason": "too_few_valid_samples",
                "success_rate": success_rate,
                "amp_x_px": np.nan,
                "amp_y_px": np.nan,
                "rms_x_px": np.nan,
                "rms_y_px": np.nan,
                "mean_x_px": np.nan,
                "mean_y_px": np.nan,
                "mean_jump_px": np.nan,
                "max_jump_px": np.nan,
                "dominant_frequency_hz": np.nan,
                "num_samples": len(d),
                "total_samples": total_count,
            })
            continue

        d = d.sort_values("time_seconds")

        d["x_rel"] = d["x"] - d["x"].iloc[0]
        d["y_rel"] = d["y"] - d["y"].iloc[0]

        amp_x = 0.5 * (d["x_rel"].max() - d["x_rel"].min())
        amp_y = 0.5 * (d["y_rel"].max() - d["y_rel"].min())

        rms_x = np.sqrt(np.mean(d["x_rel"] ** 2))
        rms_y = np.sqrt(np.mean(d["y_rel"] ** 2))

        if "jump_px" in d.columns:
            mean_jump = d["jump_px"].mean()
            max_jump = d["jump_px"].max()
        else:
            mean_jump = np.sqrt(d["dx"] ** 2 + d["dy"] ** 2).mean() if {"dx", "dy"}.issubset(d.columns) else np.nan
            max_jump = np.sqrt(d["dx"] ** 2 + d["dy"] ** 2).max() if {"dx", "dy"}.issubset(d.columns) else np.nan

        dominant_frequency_hz, freq, spectrum = compute_dominant_frequency(
            d["time_seconds"].values,
            d["y_rel"].values,
            max_freq=args.max_freq,
        )

        is_valid = (
            success_rate >= args.min_success_rate
            and (np.isnan(mean_jump) or mean_jump <= args.max_mean_jump)
            and np.isfinite(dominant_frequency_hz)
        )

        reason = "ok" if is_valid else "low_quality_tracking"

        summary.append({
            "point_id": pid,
            "valid": bool(is_valid),
            "reason": reason,
            "success_rate": success_rate,
            "amp_x_px": amp_x,
            "amp_y_px": amp_y,
            "rms_x_px": rms_x,
            "rms_y_px": rms_y,
            "mean_x_px": d["x"].mean(),
            "mean_y_px": d["y"].mean(),
            "mean_jump_px": mean_jump,
            "max_jump_px": max_jump,
            "dominant_frequency_hz": dominant_frequency_hz,
            "num_samples": len(d),
            "total_samples": total_count,
        })

        save_signal_plot(
            d["time_seconds"],
            d["x_rel"],
            os.path.join(args.out_dir, f"point_{pid}_x_displacement.png"),
            f"Point {pid}: horizontal displacement",
            "Relative x displacement [px]",
        )

        save_signal_plot(
            d["time_seconds"],
            d["y_rel"],
            os.path.join(args.out_dir, f"point_{pid}_y_displacement.png"),
            f"Point {pid}: vertical displacement",
            "Relative y displacement [px]",
        )

        save_fft_plot(
            freq,
            spectrum,
            os.path.join(args.out_dir, f"point_{pid}_fft.png"),
            f"Point {pid}: frequency spectrum",
            max_freq=args.max_freq,
        )

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.out_dir, "optical_flow_motion_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("Analyse abgeschlossen.")
    print(f"Summary: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()