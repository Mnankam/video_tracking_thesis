casefrom __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_AXES = ["linx", "liny", "linz", "rotx", "roty", "rotz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a complete IMU archive and rank measurements by temporal and "
            "spectral similarity to a reference video."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--date",
        default=None,
        help="Optional YYYY-MM-DD filter. Omit it to scan the complete archive.",
    )
    parser.add_argument("--video-start", required=True)
    parser.add_argument("--video-duration-s", required=True, type=float)
    parser.add_argument("--reference-frequency-hz", required=True, type=float)
    parser.add_argument("--axes", nargs="+", default=DEFAULT_AXES)
    parser.add_argument("--min-frequency", type=float, default=0.5)
    parser.add_argument("--max-frequency", type=float, default=50.0)
    parser.add_argument("--spectral-threshold-percent", type=float, default=10.0)
    parser.add_argument("--time-margin-s", type=float, default=180.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--number-of-peaks", type=int, default=5)
    return parser.parse_args()


def estimate_sampling_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if dt.size else float("nan")


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
    return max(0.0, float((min(end_a, end_b) - max(start_a, start_b)).total_seconds()))


def interval_gap_seconds(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    if interval_overlap_seconds(start_a, end_a, start_b, end_b) > 0:
        return 0.0
    if end_a < start_b:
        return float((start_b - end_a).total_seconds())
    return float((start_a - end_b).total_seconds())


def fft_spectrum(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_hz: float,
    maximum_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(time_s) & np.isfinite(values)
    t = np.asarray(time_s[valid], dtype=float)
    x = np.asarray(values[valid], dtype=float)

    if len(t) < 8:
        return np.array([]), np.array([])

    order = np.argsort(t)
    t, x = t[order], x[order]
    fs = estimate_sampling_rate(t)

    if not np.isfinite(fs) or fs <= 0:
        return np.array([]), np.array([])

    x = x - np.polyval(np.polyfit(t, x, 1), t)
    window = np.hanning(len(x))
    transformed = np.fft.rfft(x * window)
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / fs)

    gain = max(np.sum(window) / len(window), 1e-12)
    amplitudes = np.abs(transformed) / (len(x) * gain)
    if len(amplitudes) > 2:
        amplitudes[1:-1] *= 2.0

    upper = min(maximum_hz, fs / 2.0)
    mask = (
        (frequencies >= minimum_hz)
        & (frequencies <= upper)
        & (frequencies > 0)
    )
    return frequencies[mask], amplitudes[mask]


def strongest_local_peaks(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    number_of_peaks: int,
) -> list[tuple[float, float]]:
    if frequencies.size == 0:
        return []
    if frequencies.size < 3:
        index = int(np.argmax(amplitudes))
        return [(float(frequencies[index]), float(amplitudes[index]))]

    indices = np.where(
        (amplitudes[1:-1] > amplitudes[:-2])
        & (amplitudes[1:-1] >= amplitudes[2:])
    )[0] + 1

    if indices.size == 0:
        indices = np.array([int(np.argmax(amplitudes))])

    indices = indices[np.argsort(amplitudes[indices])[::-1]][:number_of_peaks]
    return [(float(frequencies[i]), float(amplitudes[i])) for i in indices]


def statistics(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {name: np.nan for name in (
            "mean", "std", "rms", "min", "max", "robust_amplitude"
        )}
    q05, q95 = np.percentile(valid, [5, 95])
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "rms": float(np.sqrt(np.mean(valid ** 2))),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "robust_amplitude": float(q95 - q05),
    }


def read_measurement(
    path: Path,
    axes: list[str],
    min_frequency: float,
    max_frequency: float,
    reference_frequency: float,
    video_start: pd.Timestamp,
    video_end: pd.Timestamp,
    number_of_peaks: int,
) -> tuple[dict, list[dict]]:
    result = {
        "path": str(path.resolve()),
        "folder": path.parent.name,
        "status": "ok",
    }
    peak_rows: list[dict] = []

    try:
        frame = pd.read_csv(path, sep=r"\s+")
    except Exception as exc:
        result["status"] = f"read_error: {exc}"
        return result, peak_rows

    if not {"day", "time"}.issubset(frame.columns):
        result["status"] = "missing_day_or_time"
        return result, peak_rows

    timestamps = pd.to_datetime(
        frame["day"].astype(str) + " " + frame["time"].astype(str),
        errors="coerce",
    )
    valid_time = timestamps.notna()
    frame = frame.loc[valid_time].reset_index(drop=True)
    timestamps = timestamps.loc[valid_time].reset_index(drop=True)

    if len(frame) < 8:
        result["status"] = "too_few_valid_rows"
        return result, peak_rows

    start, end = timestamps.iloc[0], timestamps.iloc[-1]
    relative_time = (timestamps - start).dt.total_seconds().to_numpy(dtype=float)
    overlap = interval_overlap_seconds(start, end, video_start, video_end)
    gap = interval_gap_seconds(start, end, video_start, video_end)

    result.update({
        "rows": int(len(frame)),
        "measurement_start": start,
        "measurement_end": end,
        "duration_s": float((end - start).total_seconds()),
        "sampling_rate_hz": estimate_sampling_rate(relative_time),
        "video_overlap_s": overlap,
        "video_overlap_fraction_percent": (
            overlap / max((video_end - video_start).total_seconds(), 1e-12) * 100.0
        ),
        "interval_gap_s": gap,
        "start_difference_s": float((start - video_start).total_seconds()),
        "end_difference_s": float((end - video_end).total_seconds()),
    })

    dominant_rows = []
    any_peak_rows = []

    for axis in axes:
        if axis not in frame.columns:
            result[f"{axis}_status"] = "missing"
            continue

        values = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
        for key, value in statistics(values).items():
            result[f"{axis}_{key}"] = value

        frequencies, amplitudes = fft_spectrum(
            relative_time, values, min_frequency, max_frequency
        )
        peaks = strongest_local_peaks(
            frequencies, amplitudes, number_of_peaks
        )

        if not peaks:
            result[f"{axis}_status"] = "no_fft_peaks"
            continue

        dominant_frequency, dominant_amplitude = peaks[0]
        dominant_error = relative_error(reference_frequency, dominant_frequency)
        dominant_rows.append(
            (dominant_error, axis, dominant_frequency, dominant_amplitude)
        )

        result[f"{axis}_dominant_frequency_hz"] = dominant_frequency
        result[f"{axis}_frequency_error_percent"] = dominant_error
        result[f"{axis}_spectral_peak_amplitude"] = dominant_amplitude

        for rank, (frequency, amplitude) in enumerate(peaks, start=1):
            error = relative_error(reference_frequency, frequency)
            relative_strength = (
                amplitude / dominant_amplitude * 100.0
                if dominant_amplitude > 0 else np.nan
            )
            any_peak_rows.append(
                (error, axis, rank, frequency, amplitude, relative_strength)
            )
            peak_rows.append({
                "path": str(path.resolve()),
                "folder": path.parent.name,
                "measurement_start": start,
                "measurement_end": end,
                "axis": axis,
                "peak_rank": rank,
                "peak_frequency_hz": frequency,
                "peak_amplitude": amplitude,
                "relative_strength_percent": relative_strength,
                "frequency_error_percent": error,
                "interval_gap_s": gap,
                "video_overlap_s": overlap,
            })

    valid_dominant = [row for row in dominant_rows if np.isfinite(row[0])]
    if valid_dominant:
        error, axis, frequency, amplitude = min(valid_dominant, key=lambda x: x[0])
        result.update({
            "best_axis": axis,
            "best_dominant_frequency_hz": frequency,
            "best_frequency_error_percent": error,
            "best_spectral_peak_amplitude": amplitude,
        })

    valid_any = [row for row in any_peak_rows if np.isfinite(row[0])]
    if valid_any:
        error, axis, rank, frequency, amplitude, strength = min(
            valid_any, key=lambda x: x[0]
        )
        result.update({
            "best_any_peak_axis": axis,
            "best_any_peak_rank": int(rank),
            "best_any_peak_frequency_hz": frequency,
            "best_any_peak_error_percent": error,
            "best_any_peak_amplitude": amplitude,
            "best_any_peak_relative_strength_percent": strength,
        })

    return result, peak_rows


def save_ranked(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str],
    ascending: list[bool],
    limit: int | None,
) -> None:
    if not frame.empty:
        frame = frame.sort_values(columns, ascending=ascending)
        if limit is not None:
            frame = frame.head(limit)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)

    if not root.is_dir():
        raise NotADirectoryError(f"Messdatenordner fehlt: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    video_start = pd.Timestamp(args.video_start)
    video_end = video_start + pd.to_timedelta(args.video_duration_s, unit="s")

    files = sorted(root.rglob("data.csv"))
    if args.date:
        files = [path for path in files if args.date in str(path)]

    if not files:
        raise FileNotFoundError("Keine data.csv gefunden.")

    scope = f"date filter {args.date}" if args.date else "complete archive"
    print(f"Scan scope: {scope}")
    print(f"Found data.csv files: {len(files)}")
    print(f"Video interval: {video_start} to {video_end}")
    print(f"Reference frequency: {args.reference_frequency_hz:.6f} Hz")

    rows, peaks = [], []
    for index, path in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {path}")
        row, peak_rows = read_measurement(
            path,
            args.axes,
            args.min_frequency,
            args.max_frequency,
            args.reference_frequency_hz,
            video_start,
            video_end,
            args.number_of_peaks,
        )
        rows.append(row)
        peaks.extend(peak_rows)

    inventory = pd.DataFrame(rows)
    peak_inventory = pd.DataFrame(peaks)
    inventory.to_csv(out_dir / "measurement_inventory.csv", index=False)
    peak_inventory.to_csv(out_dir / "fft_peak_inventory.csv", index=False)

    valid = inventory[inventory["status"] == "ok"].copy()
    failed = inventory[inventory["status"] != "ok"].copy()
    failed.to_csv(out_dir / "failed_measurements.csv", index=False)

    temporal = valid[
        (valid["video_overlap_s"] > 0)
        | (valid["interval_gap_s"] <= args.time_margin_s)
    ].copy()

    dominant = valid[
        valid["best_frequency_error_percent"]
        <= args.spectral_threshold_percent
    ].copy()

    top_peak = valid[
        valid["best_any_peak_error_percent"]
        <= args.spectral_threshold_percent
    ].copy()

    simultaneous = valid[
        (
            (valid["video_overlap_s"] > 0)
            | (valid["interval_gap_s"] <= args.time_margin_s)
        )
        & (
            valid["best_any_peak_error_percent"]
            <= args.spectral_threshold_percent
        )
    ].copy()

    save_ranked(
        temporal,
        out_dir / "temporal_candidates.csv",
        ["video_overlap_s", "interval_gap_s", "best_any_peak_error_percent"],
        [False, True, True],
        args.top_n,
    )
    save_ranked(
        dominant,
        out_dir / "dominant_frequency_candidates.csv",
        ["best_frequency_error_percent", "interval_gap_s"],
        [True, True],
        args.top_n,
    )
    save_ranked(
        top_peak,
        out_dir / "top_peak_frequency_candidates.csv",
        ["best_any_peak_error_percent", "best_any_peak_rank", "interval_gap_s"],
        [True, True, True],
        args.top_n,
    )
    save_ranked(
        simultaneous,
        out_dir / "simultaneous_candidates.csv",
        ["video_overlap_s", "interval_gap_s", "best_any_peak_error_percent"],
        [False, True, True],
        None,
    )

    combined = valid.copy()
    combined["time_score"] = combined["interval_gap_s"] / max(
        args.video_duration_s, 1.0
    )
    combined["frequency_score"] = (
        combined["best_any_peak_error_percent"] / 100.0
    )
    combined["combined_exploratory_score"] = (
        combined["time_score"] + combined["frequency_score"]
    )
    save_ranked(
        combined,
        out_dir / "combined_candidates.csv",
        ["combined_exploratory_score", "interval_gap_s", "best_any_peak_error_percent"],
        [True, True, True],
        args.top_n,
    )

    conclusion = (
        "At least one measurement satisfies both temporal and spectral criteria."
        if len(simultaneous)
        else (
            "No scanned measurement satisfies both temporal and spectral "
            "criteria under the configured thresholds."
        )
    )

    summary = {
        "root": str(root.resolve()),
        "scan_scope": scope,
        "date_filter": args.date,
        "number_of_files_scanned": int(len(files)),
        "number_successfully_read": int(len(valid)),
        "number_failed": int(len(failed)),
        "video_start": str(video_start),
        "video_end": str(video_end),
        "video_duration_s": float(args.video_duration_s),
        "reference_frequency_hz": float(args.reference_frequency_hz),
        "time_margin_s": float(args.time_margin_s),
        "spectral_threshold_percent": float(args.spectral_threshold_percent),
        "number_of_fft_peaks_per_axis": int(args.number_of_peaks),
        "number_temporal_candidates": int(len(temporal)),
        "number_dominant_frequency_candidates": int(len(dominant)),
        "number_top_peak_frequency_candidates": int(len(top_peak)),
        "number_simultaneous_candidates": int(len(simultaneous)),
        "conclusion": conclusion,
        "important_limitation": (
            "A spectral match alone does not prove experimental pairing. "
            "Metadata or supervisor confirmation remains necessary."
        ),
    }

    pd.DataFrame([summary]).to_csv(
        out_dir / "search_summary.csv", index=False
    )
    with (out_dir / "search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\nArchive search completed.")
    print(f"Files scanned: {len(files)}")
    print(f"Successfully read: {len(valid)}")
    print(f"Failed: {len(failed)}")
    print(f"Temporal candidates: {len(temporal)}")
    print(f"Dominant-frequency candidates: {len(dominant)}")
    print(f"Top-{args.number_of_peaks} peak candidates: {len(top_peak)}")
    print(f"Simultaneous candidates: {len(simultaneous)}")
    print(f"Conclusion: {conclusion}")
    print(f"Results directory: {out_dir}")


if __name__ == "__main__":
    main()
