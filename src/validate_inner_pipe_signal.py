from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_signal(
    df: pd.DataFrame,
    time_column: str,
    signal_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read, clean, sort and deduplicate the selected time series."""
    data = df[[time_column, signal_column]].copy()

    data[time_column] = pd.to_numeric(data[time_column], errors="coerce")
    data[signal_column] = pd.to_numeric(data[signal_column], errors="coerce")
    data = data.dropna()
    data = data.sort_values(time_column)

    # At most one value per timestamp.
    data = data.groupby(time_column, as_index=False)[signal_column].mean()

    if len(data) < 8:
        raise ValueError("Zu wenige gültige Messpunkte für die Signalanalyse.")

    time = data[time_column].to_numpy(dtype=float)
    signal = data[signal_column].to_numpy(dtype=float)

    return time, signal


def estimate_sampling_rate(time: np.ndarray) -> float:
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        raise ValueError("Die Abtastzeit konnte nicht bestimmt werden.")

    return float(1.0 / np.median(dt))


def detrend_linear(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    coefficients = np.polyfit(time, signal, deg=1)
    trend = np.polyval(coefficients, time)
    return signal - trend


def dominant_frequency(
    time: np.ndarray,
    signal: np.ndarray,
    min_frequency: float,
    max_frequency: float | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    fs = estimate_sampling_rate(time)
    centered = detrend_linear(time, signal)

    # Hann window reduces spectral leakage.
    window = np.hanning(len(centered))
    windowed = centered * window

    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(len(windowed), d=1.0 / fs)
    magnitude = np.abs(spectrum)

    valid = frequencies >= min_frequency

    if max_frequency is not None:
        valid &= frequencies <= max_frequency

    # DC component must not be selected.
    valid &= frequencies > 0

    if not np.any(valid):
        raise ValueError("Kein gültiger Frequenzbereich für die FFT vorhanden.")

    valid_indices = np.where(valid)[0]
    peak_index = valid_indices[np.argmax(magnitude[valid])]

    return float(frequencies[peak_index]), frequencies, magnitude


def estimate_amplitude(signal: np.ndarray) -> float:
    """Robust half peak-to-peak amplitude in pixels."""
    low = np.percentile(signal, 5)
    high = np.percentile(signal, 95)
    return float((high - low) / 2.0)


def downsample_signal(
    time: np.ndarray,
    signal: np.ndarray,
    original_fs: float,
    target_fs: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if target_fs > original_fs:
        raise ValueError(
            f"Zielabtastrate {target_fs} Hz ist größer als "
            f"die Originalabtastrate {original_fs:.3f} Hz."
        )

    step = max(1, int(round(original_fs / target_fs)))

    ds_time = time[::step]
    ds_signal = signal[::step]
    actual_fs = original_fs / step

    return ds_time, ds_signal, float(actual_fs)


def relative_error(reference: float, value: float) -> float:
    if reference == 0:
        return float("nan")

    return float(abs(value - reference) / abs(reference) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the tracked inner-pipe motion signal."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument(
        "--time-column",
        default="time_seconds",
    )
    parser.add_argument(
        "--signal-column",
        default="inner_pipe_track_center_x",
    )

    parser.add_argument(
        "--target-fps",
        nargs="+",
        type=float,
        default=[200.0, 100.0, 50.0, 25.0, 20.0, 10.0],
    )

    parser.add_argument(
        "--min-frequency",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    time, signal = prepare_signal(
        df,
        args.time_column,
        args.signal_column,
    )

    original_fs = estimate_sampling_rate(time)
    duration = float(time[-1] - time[0])
    frequency_resolution = 1.0 / duration if duration > 0 else float("nan")

    original_frequency, frequencies, magnitude = dominant_frequency(
        time,
        signal,
        args.min_frequency,
        args.max_frequency,
    )
    original_amplitude = estimate_amplitude(signal)

    rows: list[dict[str, float | int]] = []

    for requested_fs in args.target_fps:
        if requested_fs > original_fs + 1e-6:
            print(
                f"Übersprungen: {requested_fs} Hz > "
                f"Originalabtastrate {original_fs:.3f} Hz"
            )
            continue

        ds_time, ds_signal, actual_fs = downsample_signal(
            time,
            signal,
            original_fs,
            requested_fs,
        )

        if len(ds_signal) < 8:
            print(
                f"Übersprungen: {actual_fs:.3f} Hz liefert "
                f"nur {len(ds_signal)} Punkte."
            )
            continue

        max_allowed_frequency = args.max_frequency

        # Nyquist limit for this downsampled signal.
        nyquist_frequency = actual_fs / 2.0

        if max_allowed_frequency is None:
            analysis_max_frequency = nyquist_frequency
        else:
            analysis_max_frequency = min(
                max_allowed_frequency,
                nyquist_frequency,
            )

        frequency, _, _ = dominant_frequency(
            ds_time,
            ds_signal,
            args.min_frequency,
            analysis_max_frequency,
        )
        amplitude = estimate_amplitude(ds_signal)

        rows.append(
            {
                "requested_sampling_rate_hz": requested_fs,
                "actual_sampling_rate_hz": actual_fs,
                "nyquist_frequency_hz": nyquist_frequency,
                "number_of_samples": len(ds_signal),
                "duration_s": float(ds_time[-1] - ds_time[0]),
                "dominant_frequency_hz": frequency,
                "frequency_relative_error_percent": relative_error(
                    original_frequency,
                    frequency,
                ),
                "amplitude_px": amplitude,
                "amplitude_relative_error_percent": relative_error(
                    original_amplitude,
                    amplitude,
                ),
                "nyquist_condition_satisfied": int(
                    actual_fs >= 2.0 * original_frequency
                ),
            }
        )

    results = pd.DataFrame(rows)

    results_path = os.path.join(
        args.out_dir,
        "downsampling_validation.csv",
    )
    results.to_csv(results_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "input_csv": args.csv,
                "signal_column": args.signal_column,
                "number_of_samples": len(signal),
                "duration_s": duration,
                "original_sampling_rate_hz": original_fs,
                "frequency_resolution_hz": frequency_resolution,
                "original_dominant_frequency_hz": original_frequency,
                "original_amplitude_px": original_amplitude,
                "signal_min_px": float(np.min(signal)),
                "signal_max_px": float(np.max(signal)),
                "signal_mean_px": float(np.mean(signal)),
                "signal_std_px": float(np.std(signal, ddof=1)),
            }
        ]
    )

    summary_path = os.path.join(
        args.out_dir,
        "validation_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    # Time signal.
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)
    plt.xlabel("Time [s]")
    plt.ylabel("Inner pipe position x [pixel]")
    plt.title("GX010044 - Inner pipe tracking signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, "inner_pipe_timeseries.png"),
        dpi=300,
    )
    plt.close()

    # FFT spectrum.
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, magnitude)
    plt.axvline(
        original_frequency,
        linestyle="--",
        label=f"Dominant frequency = {original_frequency:.3f} Hz",
    )
    plt.xlim(
        left=0,
        right=args.max_frequency
        if args.max_frequency is not None
        else original_fs / 2.0,
    )
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("FFT magnitude")
    plt.title("Frequency spectrum of the tracked inner-pipe motion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, "inner_pipe_fft.png"),
        dpi=300,
    )
    plt.close()

    # Downsampling comparison.
    if not results.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(
            results["actual_sampling_rate_hz"],
            results["dominant_frequency_hz"],
            marker="o",
        )
        plt.axhline(
            original_frequency,
            linestyle="--",
            label=f"Reference = {original_frequency:.3f} Hz",
        )
        plt.xlabel("Sampling rate [Hz]")
        plt.ylabel("Dominant frequency [Hz]")
        plt.title("Influence of temporal downsampling")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.out_dir,
                "downsampling_frequency_comparison.png",
            ),
            dpi=300,
        )
        plt.close()

    print("Validation completed.")
    print(f"Input samples: {len(signal)}")
    print(f"Duration: {duration:.6f} s")
    print(f"Sampling rate: {original_fs:.6f} Hz")
    print(f"Frequency resolution: {frequency_resolution:.6f} Hz")
    print(f"Dominant frequency: {original_frequency:.6f} Hz")
    print(f"Amplitude: {original_amplitude:.6f} px")
    print(f"Summary: {summary_path}")
    print(f"Downsampling results: {results_path}")


if __name__ == "__main__":
    main()
