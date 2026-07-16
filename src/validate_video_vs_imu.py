from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AXES = ["linx", "liny", "linz", "rotx", "roty", "rotz"]


def sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Abtastrate konnte nicht bestimmt werden.")
    return float(1.0 / np.median(dt))


def prepare_signal(time_s: np.ndarray, values: np.ndarray):
    valid = np.isfinite(time_s) & np.isfinite(values)
    time_s = np.asarray(time_s[valid], dtype=float)
    values = np.asarray(values[valid], dtype=float)

    order = np.argsort(time_s)
    time_s = time_s[order]
    values = values[order]

    if len(time_s) < 8:
        raise ValueError("Zu wenige gültige Messpunkte.")

    return time_s, values, sampling_rate(time_s)


def spectrum(time_s: np.ndarray, values: np.ndarray):
    fs = sampling_rate(time_s)

    coefficients = np.polyfit(time_s, values, 1)
    detrended = values - np.polyval(coefficients, time_s)

    window = np.hanning(len(detrended))
    transformed = np.fft.rfft(detrended * window)
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0 / fs)

    coherent_gain = np.sum(window) / len(window)
    amplitudes = np.abs(transformed) / (len(detrended) * coherent_gain)
    if len(amplitudes) > 2:
        amplitudes[1:-1] *= 2.0

    return frequencies, amplitudes


def dominant_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_hz: float,
    maximum_hz: float,
):
    frequencies, amplitudes = spectrum(time_s, values)
    fs = sampling_rate(time_s)
    upper = min(maximum_hz, fs / 2.0)

    valid = (
        (frequencies >= minimum_hz)
        & (frequencies <= upper)
        & (frequencies > 0)
    )
    if not valid.any():
        raise ValueError("Kein gültiger FFT-Frequenzbereich.")

    indices = np.where(valid)[0]
    peak_index = indices[np.argmax(amplitudes[valid])]

    return (
        float(frequencies[peak_index]),
        float(amplitudes[peak_index]),
        frequencies,
        amplitudes,
    )


def relative_error(reference: float, value: float) -> float:
    if np.isclose(reference, 0.0):
        return float("nan")
    return float(abs(value - reference) / abs(reference) * 100.0)


def robust_amplitude(values: np.ndarray) -> float:
    return float(
        (np.percentile(values, 95) - np.percentile(values, 5)) / 2.0
    )


def load_video(path: Path, time_column: str, signal_column: str):
    frame = pd.read_csv(path)
    for column in (time_column, signal_column):
        if column not in frame.columns:
            raise KeyError(f"Spalte fehlt in Video-CSV: {column}")

    time_s = pd.to_numeric(frame[time_column], errors="coerce").to_numpy()
    values = pd.to_numeric(frame[signal_column], errors="coerce").to_numpy()
    return prepare_signal(time_s, values)


def load_imu(path: Path, day_column: str, time_column: str):
    frame = pd.read_csv(path, sep=r"\s+")

    for column in (day_column, time_column):
        if column not in frame.columns:
            raise KeyError(f"Spalte fehlt in IMU-CSV: {column}")

    timestamps = pd.to_datetime(
        frame[day_column].astype(str)
        + " "
        + frame[time_column].astype(str),
        errors="coerce",
    )

    valid = timestamps.notna()
    frame = frame.loc[valid].reset_index(drop=True)
    timestamps = timestamps.loc[valid].reset_index(drop=True)

    if len(frame) < 8:
        raise ValueError("Zu wenige gültige IMU-Zeitstempel.")

    return frame, timestamps


def plot_spectrum(
    frequencies,
    amplitudes,
    peak_frequency,
    title,
    output,
    maximum_hz,
):
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, amplitudes)
    plt.axvline(
        peak_frequency,
        linestyle="--",
        label=f"Dominant frequency = {peak_frequency:.4f} Hz",
    )
    plt.xlim(0, maximum_hz)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Single-sided amplitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 of video-vs-IMU validation: FFT analysis of all "
            "IMU axes and dominant-frequency comparison."
        )
    )
    parser.add_argument("--video-csv", required=True)
    parser.add_argument("--imu-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--video-time-column", default="time_seconds")
    parser.add_argument(
        "--video-signal-column",
        default="inner_pipe_track_center_x",
    )
    parser.add_argument("--imu-day-column", default="day")
    parser.add_argument("--imu-time-column", default="time")
    parser.add_argument("--imu-axes", nargs="+", default=DEFAULT_AXES)
    parser.add_argument("--min-frequency", type=float, default=0.5)
    parser.add_argument("--max-frequency", type=float, default=50.0)
    parser.add_argument(
        "--video-reference-frequency",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--acceptance-threshold-percent",
        type=float,
        default=10.0,
    )
    args = parser.parse_args()

    video_path = Path(args.video_csv)
    imu_path = Path(args.imu_csv)
    output_dir = Path(args.out_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video-CSV fehlt: {video_path}")
    if not imu_path.is_file():
        raise FileNotFoundError(f"IMU-CSV fehlt: {imu_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_time, video_values, video_fs = load_video(
        video_path,
        args.video_time_column,
        args.video_signal_column,
    )
    (
        estimated_video_frequency,
        video_peak_amplitude,
        video_frequencies,
        video_amplitudes,
    ) = dominant_frequency(
        video_time,
        video_values,
        args.min_frequency,
        args.max_frequency,
    )

    video_frequency = (
        args.video_reference_frequency
        if args.video_reference_frequency is not None
        else estimated_video_frequency
    )

    plot_spectrum(
        video_frequencies,
        video_amplitudes,
        video_frequency,
        "Video-derived inner-pipe frequency spectrum",
        output_dir / "video_fft.png",
        min(args.max_frequency, video_fs / 2.0),
    )

    imu_frame, imu_timestamps = load_imu(
        imu_path,
        args.imu_day_column,
        args.imu_time_column,
    )
    imu_time = (
        imu_timestamps - imu_timestamps.iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float)

    rows = []
    overview_signals = {}

    for axis in args.imu_axes:
        if axis not in imu_frame.columns:
            print(f"Warnung: Achse fehlt und wird übersprungen: {axis}")
            continue

        values = pd.to_numeric(
            imu_frame[axis],
            errors="coerce",
        ).to_numpy(dtype=float)

        time_s, values, axis_fs = prepare_signal(imu_time, values)
        overview_signals[axis] = (time_s, values)

        (
            axis_frequency,
            axis_peak_amplitude,
            frequencies,
            amplitudes,
        ) = dominant_frequency(
            time_s,
            values,
            args.min_frequency,
            args.max_frequency,
        )

        error = relative_error(video_frequency, axis_frequency)

        rows.append(
            {
                "axis": axis,
                "number_of_samples": len(values),
                "duration_s": float(time_s[-1] - time_s[0]),
                "sampling_rate_hz": axis_fs,
                "dominant_frequency_hz": axis_frequency,
                "frequency_relative_error_percent": error,
                "spectral_peak_amplitude": axis_peak_amplitude,
                "robust_time_amplitude": robust_amplitude(values),
                "within_acceptance_threshold": int(
                    error <= args.acceptance_threshold_percent
                ),
            }
        )

        plot_spectrum(
            frequencies,
            amplitudes,
            axis_frequency,
            f"IMU frequency spectrum: {axis}",
            output_dir / f"imu_fft_{axis}.png",
            min(args.max_frequency, axis_fs / 2.0),
        )

    if not rows:
        raise ValueError("Keine gültige IMU-Achse konnte ausgewertet werden.")

    results = pd.DataFrame(rows).sort_values(
        "frequency_relative_error_percent"
    )
    best_axis = str(results.iloc[0]["axis"])
    results["selected_as_best_axis"] = (
        results["axis"] == best_axis
    ).astype(int)
    results.to_csv(output_dir / "frequency_comparison.csv", index=False)

    best = results.iloc[0]
    best_summary = pd.DataFrame(
        [
            {
                "best_axis": best_axis,
                "video_dominant_frequency_hz": video_frequency,
                "imu_dominant_frequency_hz": best[
                    "dominant_frequency_hz"
                ],
                "frequency_relative_error_percent": best[
                    "frequency_relative_error_percent"
                ],
                "acceptance_threshold_percent": (
                    args.acceptance_threshold_percent
                ),
                "frequency_validation_passed": int(
                    best["frequency_relative_error_percent"]
                    <= args.acceptance_threshold_percent
                ),
            }
        ]
    )
    best_summary.to_csv(output_dir / "best_axis.csv", index=False)

    figure, axes = plt.subplots(
        len(overview_signals),
        1,
        figsize=(14, 2.4 * len(overview_signals)),
        sharex=True,
    )
    if len(overview_signals) == 1:
        axes = [axes]

    for plot_axis, (axis, signal) in zip(
        axes,
        overview_signals.items(),
    ):
        time_s, values = signal
        plot_axis.plot(time_s, values, linewidth=0.7)
        plot_axis.set_ylabel(axis)
        plot_axis.grid(True)

    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("IMU signal overview")
    plt.tight_layout()
    plt.savefig(output_dir / "imu_overview.png", dpi=300)
    plt.close()

    ordered = results.sort_values("frequency_relative_error_percent")
    positions = np.arange(len(ordered))

    plt.figure(figsize=(10, 5))
    plt.bar(positions, ordered["dominant_frequency_hz"])
    plt.axhline(
        video_frequency,
        linestyle="--",
        label=f"Video reference = {video_frequency:.4f} Hz",
    )
    plt.xticks(positions, ordered["axis"])
    plt.xlabel("IMU axis")
    plt.ylabel("Dominant frequency [Hz]")
    plt.title("Dominant-frequency comparison")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        positions,
        ordered["frequency_relative_error_percent"],
    )
    plt.axhline(
        args.acceptance_threshold_percent,
        linestyle="--",
        label=(
            "Acceptance threshold = "
            f"{args.acceptance_threshold_percent:.1f}%"
        ),
    )
    plt.xticks(positions, ordered["axis"])
    plt.xlabel("IMU axis")
    plt.ylabel("Relative frequency error [%]")
    plt.title("Relative dominant-frequency error")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_error.png", dpi=300)
    plt.close()

    summary = {
        "video_csv": str(video_path),
        "imu_csv": str(imu_path),
        "video_number_of_samples": int(len(video_values)),
        "video_duration_s": float(video_time[-1] - video_time[0]),
        "video_sampling_rate_hz": float(video_fs),
        "video_estimated_dominant_frequency_hz": float(
            estimated_video_frequency
        ),
        "video_reference_frequency_hz": float(video_frequency),
        "video_spectral_peak_amplitude": float(video_peak_amplitude),
        "imu_measurement_start": str(imu_timestamps.iloc[0]),
        "imu_measurement_end": str(imu_timestamps.iloc[-1]),
        "imu_number_of_rows": int(len(imu_frame)),
        "best_axis": best_axis,
        "best_axis_dominant_frequency_hz": float(
            best["dominant_frequency_hz"]
        ),
        "best_axis_frequency_relative_error_percent": float(
            best["frequency_relative_error_percent"]
        ),
        "frequency_validation_passed": bool(
            best["frequency_relative_error_percent"]
            <= args.acceptance_threshold_percent
        ),
        "limitation": (
            "Stage 1 validates dominant-frequency agreement only. "
            "Synchronization, waveform correlation, amplitude agreement, "
            "and damping agreement are not yet evaluated."
        ),
    }

    with open(
        output_dir / "validation_stage1_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    pd.DataFrame([summary]).to_csv(
        output_dir / "validation_stage1_summary.csv",
        index=False,
    )

    print("Stage-1 video-vs-IMU validation completed.")
    print(f"Video sampling rate: {video_fs:.6f} Hz")
    print(f"Video dominant frequency: {video_frequency:.6f} Hz")
    print(f"IMU measurement start: {imu_timestamps.iloc[0]}")
    print(f"IMU measurement end: {imu_timestamps.iloc[-1]}")
    print(f"Selected IMU axis: {best_axis}")
    print(
        "Selected IMU dominant frequency: "
        f"{float(best['dominant_frequency_hz']):.6f} Hz"
    )
    print(
        "Relative frequency error: "
        f"{float(best['frequency_relative_error_percent']):.6f}%"
    )
    print(
        "Frequency validation passed: "
        f"{summary['frequency_validation_passed']}"
    )
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
