from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_AXES = ["linx", "liny", "linz", "rotx", "roty", "rotz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a measurement archive for plausible IMU files belonging "
            "to a video. The script inventories timestamps, duration, sampling "
            "rate and dominant frequencies for all data.csv files."
        )
    )
    parser.add_argument("--root", required=True, help="Root folder containing data.csv files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--date",
        default=None,
        help="Optional date filter in YYYY-MM-DD format, e.g. 2022-02-25.",
    )
    parser.add_argument(
        "--video-start",
        required=True,
        help="Video start timestamp, e.g. '2022-02-25 16:34:23'.",
    )
    parser.add_argument("--video-duration-s", required=True, type=float)
    parser.add_argument("--reference-frequency-hz", required=True, type=float)
    parser.add_argument("--axes", nargs="+", default=DEFAULT_AXES)
    parser.add_argument("--min-frequency", type=float, default=0.5)
    parser.add_argument("--max-frequency", type=float, default=50.0)
    parser.add_argument(
        "--spectral-threshold-percent",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--time-margin-s",
        type=float,
        default=120.0,
        help="Additional time margin around the video interval for temporal candidates.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top candidates written to compact candidate tables.",
    )
    return parser.parse_args()


def estimate_sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def dominant_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_hz: float,
    maximum_hz: float,
) -> tuple[float, float]:
    valid = np.isfinite(time_s) & np.isfinite(values)
    t = np.asarray(time_s[valid], dtype=float)
    x = np.asarray(values[valid], dtype=float)

    if len(t) < 8:
        return float("nan"), float("nan")

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    fs = estimate_sampling_rate(t)
    if not np.isfinite(fs) or fs <= 0:
        return float("nan"), float("nan")

    # Linear detrending and Hann window to reduce leakage.
    coefficients = np.polyfit(t, x, 1)
    x = x - np.polyval(coefficients, t)
    window = np.hanning(len(x))
    transformed = np.fft.rfft(x * window)
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / fs)

    coherent_gain = np.sum(window) / len(window)
    amplitudes = np.abs(transformed) / (len(x) * coherent_gain)
    if len(amplitudes) > 2:
        amplitudes[1:-1] *= 2.0

    upper = min(maximum_hz, fs / 2.0)
    mask = (
        (frequencies >= minimum_hz)
        & (frequencies <= upper)
        & (frequencies > 0)
    )

    if not np.any(mask):
        return float("nan"), float("nan")

    indices = np.where(mask)[0]
    peak_index = indices[np.argmax(amplitudes[mask])]

    return float(frequencies[peak_index]), float(amplitudes[peak_index])


def relative_error(reference: float, value: float) -> float:
    if not np.isfinite(value) or np.isclose(reference, 0.0):
        return float("nan")
    return float(abs(value - reference) / abs(reference) * 100.0)


def interval_overlap_seconds(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    return max(0.0, float((end - start).total_seconds()))


def interval_gap_seconds(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    overlap = interval_overlap_seconds(start_a, end_a, start_b, end_b)
    if overlap > 0:
        return 0.0
    if end_a < start_b:
        return float((start_b - end_a).total_seconds())
    return float((start_a - end_b).total_seconds())


def read_measurement(
    path: Path,
    axes: list[str],
    min_frequency: float,
    max_frequency: float,
    reference_frequency: float,
    video_start: pd.Timestamp,
    video_end: pd.Timestamp,
) -> dict:
    result: dict = {
        "path": str(path.resolve()),
        "folder": path.parent.name,
        "status": "ok",
    }

    try:
        frame = pd.read_csv(path, sep=r"\s+")
    except Exception as exc:
        result["status"] = f"read_error: {exc}"
        return result

    if "day" not in frame.columns or "time" not in frame.columns:
        result["status"] = "missing_day_or_time"
        return result

    timestamps = pd.to_datetime(
        frame["day"].astype(str) + " " + frame["time"].astype(str),
        errors="coerce",
    )
    valid_time = timestamps.notna()

    frame = frame.loc[valid_time].reset_index(drop=True)
    timestamps = timestamps.loc[valid_time].reset_index(drop=True)

    if len(frame) < 8:
        result["status"] = "too_few_valid_rows"
        return result

    measurement_start = timestamps.iloc[0]
    measurement_end = timestamps.iloc[-1]
    relative_time = (
        timestamps - measurement_start
    ).dt.total_seconds().to_numpy(dtype=float)

    fs = estimate_sampling_rate(relative_time)
    duration = float((measurement_end - measurement_start).total_seconds())
    overlap = interval_overlap_seconds(
        measurement_start,
        measurement_end,
        video_start,
        video_end,
    )
    gap = interval_gap_seconds(
        measurement_start,
        measurement_end,
        video_start,
        video_end,
    )

    result.update(
        {
            "rows": int(len(frame)),
            "measurement_start": measurement_start,
            "measurement_end": measurement_end,
            "duration_s": duration,
            "sampling_rate_hz": fs,
            "video_overlap_s": overlap,
            "video_overlap_fraction_percent": (
                overlap / max((video_end - video_start).total_seconds(), 1e-12) * 100.0
            ),
            "interval_gap_s": gap,
            "start_difference_s": float(
                (measurement_start - video_start).total_seconds()
            ),
            "end_difference_s": float(
                (measurement_end - video_end).total_seconds()
            ),
        }
    )

    axis_rows = []
    for axis in axes:
        if axis not in frame.columns:
            result[f"{axis}_status"] = "missing"
            continue

        numeric = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
        frequency, amplitude = dominant_frequency(
            relative_time,
            numeric,
            min_frequency,
            max_frequency,
        )
        error = relative_error(reference_frequency, frequency)

        result[f"{axis}_dominant_frequency_hz"] = frequency
        result[f"{axis}_frequency_error_percent"] = error
        result[f"{axis}_spectral_peak_amplitude"] = amplitude

        axis_rows.append((axis, frequency, error, amplitude))

    valid_axes = [
        row for row in axis_rows
        if np.isfinite(row[2])
    ]

    if valid_axes:
        best_axis, best_frequency, best_error, best_amplitude = min(
            valid_axes,
            key=lambda row: row[2],
        )
        result.update(
            {
                "best_axis": best_axis,
                "best_dominant_frequency_hz": best_frequency,
                "best_frequency_error_percent": best_error,
                "best_spectral_peak_amplitude": best_amplitude,
            }
        )
    else:
        result.update(
            {
                "best_axis": "",
                "best_dominant_frequency_hz": np.nan,
                "best_frequency_error_percent": np.nan,
                "best_spectral_peak_amplitude": np.nan,
            }
        )

    return result


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    output_dir = Path(args.out_dir)

    if not root.is_dir():
        raise NotADirectoryError(f"Messdatenordner fehlt: {root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_start = pd.Timestamp(args.video_start)
    video_end = video_start + pd.to_timedelta(
        args.video_duration_s,
        unit="s",
    )

    files = sorted(root.rglob("data.csv"))
    if args.date:
        files = [
            path for path in files
            if args.date in str(path)
        ]

    if not files:
        raise FileNotFoundError("Keine passende data.csv gefunden.")

    print(f"Gefundene data.csv-Dateien: {len(files)}")
    print(f"Video interval: {video_start} to {video_end}")
    print(f"Reference frequency: {args.reference_frequency_hz:.6f} Hz")

    rows = []
    total = len(files)

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{total}] {path}")
        rows.append(
            read_measurement(
                path=path,
                axes=args.axes,
                min_frequency=args.min_frequency,
                max_frequency=args.max_frequency,
                reference_frequency=args.reference_frequency_hz,
                video_start=video_start,
                video_end=video_end,
            )
        )

    inventory = pd.DataFrame(rows)
    inventory_path = output_dir / "measurement_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    valid = inventory[inventory["status"] == "ok"].copy()

    temporal_candidates = valid[
        (valid["video_overlap_s"] > 0)
        | (valid["interval_gap_s"] <= args.time_margin_s)
    ].copy()
    temporal_candidates = temporal_candidates.sort_values(
        [
            "video_overlap_s",
            "interval_gap_s",
            "best_frequency_error_percent",
        ],
        ascending=[False, True, True],
    )
    temporal_candidates.head(args.top_n).to_csv(
        output_dir / "temporal_candidates.csv",
        index=False,
    )

    spectral_candidates = valid[
        valid["best_frequency_error_percent"]
        <= args.spectral_threshold_percent
    ].copy()
    spectral_candidates = spectral_candidates.sort_values(
        [
            "best_frequency_error_percent",
            "interval_gap_s",
        ],
        ascending=[True, True],
    )
    spectral_candidates.head(args.top_n).to_csv(
        output_dir / "spectral_candidates.csv",
        index=False,
    )

    # Combined ranking is exploratory only. It must not be treated as proof
    # that a measurement belongs to the video.
    combined = valid.copy()
    duration_scale = max(args.video_duration_s, 1.0)
    combined["time_score"] = (
        combined["interval_gap_s"] / duration_scale
    )
    combined["frequency_score"] = (
        combined["best_frequency_error_percent"] / 100.0
    )
    combined["combined_exploratory_score"] = (
        combined["time_score"] + combined["frequency_score"]
    )
    combined = combined.sort_values(
        [
            "combined_exploratory_score",
            "interval_gap_s",
            "best_frequency_error_percent",
        ]
    )
    combined.head(args.top_n).to_csv(
        output_dir / "combined_candidates.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "root": str(root.resolve()),
                "date_filter": args.date or "",
                "number_of_files_scanned": len(files),
                "number_successfully_read": int((inventory["status"] == "ok").sum()),
                "video_start": video_start,
                "video_end": video_end,
                "video_duration_s": args.video_duration_s,
                "reference_frequency_hz": args.reference_frequency_hz,
                "time_margin_s": args.time_margin_s,
                "spectral_threshold_percent": args.spectral_threshold_percent,
                "number_temporal_candidates": len(temporal_candidates),
                "number_spectral_candidates": len(spectral_candidates),
                "important_limitation": (
                    "Temporal proximity and frequency similarity identify candidates "
                    "only. Experimental metadata or supervisor confirmation is still "
                    "required before a file is accepted as the matching reference."
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "search_summary.csv", index=False)

    print("\nSearch completed.")
    print(f"Inventory: {inventory_path}")
    print(
        "Temporal candidates: "
        f"{output_dir / 'temporal_candidates.csv'}"
    )
    print(
        "Spectral candidates: "
        f"{output_dir / 'spectral_candidates.csv'}"
    )
    print(
        "Combined exploratory candidates: "
        f"{output_dir / 'combined_candidates.csv'}"
    )
    print(
        "\nImportant: A candidate is not automatically the correct measurement. "
        "Confirm the experimental pairing before final validation."
    )


if __name__ == "__main__":
    main()
