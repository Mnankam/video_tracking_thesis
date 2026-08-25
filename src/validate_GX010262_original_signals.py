#!/usr/bin/env python3
"""
validate_GX010262_original_signals.py

External validation for GX010262 using the original Lucas-Kanade displacement
signals. No numerical differentiation is applied.

Video signals:
- video_x_displacement_px
- video_y_displacement_px
- video_xy_displacement_px = sqrt(x^2 + y^2)

IMU signals:
- linx
- liny
- linz
- linxy_magnitude = sqrt(linx^2 + liny^2)

The script evaluates raw and 4.5-5.5 Hz band-limited waveform similarity.
Because video displacement [px] and IMU acceleration [native IMU unit] are
different physical quantities, only normalized signal-shape similarity is
interpreted; absolute amplitude agreement is not claimed.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, filtfilt

from src.validation.imu_loader import load_imu_signal


@dataclass(frozen=True)
class ValidationConfig:
    video_id: str
    video_path: Path
    imu_path: Path
    output_dir: Path
    video_time_column: str
    video_x_column: str
    video_y_column: str
    imu_axes: tuple[str, ...]
    target_sampling_rate_hz: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int
    maximum_lag_s: float
    detailed_window_start_s: float
    detailed_window_end_s: float


CONFIG = ValidationConfig(
    video_id="GX010262",
    video_path=Path(
        "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
        "video_tracking_thesis/outputs/Lucas_Kanade_CPU_1/"
        "GX010262_lucas_kanade_timeseries.csv"
    ),
    imu_path=Path(
        "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
        "video_tracking_thesis/data/archives/measured_data/"
        "2022-04-16/2022-04-16_19.43.22/data.csv"
    ),
    output_dir=Path(
        "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
        "video_tracking_thesis/outputs/external_validation/"
        "GX010262_original_signals"
    ),
    video_time_column="time_seconds",
    video_x_column="lk_displacement_x",
    video_y_column="lk_displacement_y",
    imu_axes=("linx", "liny", "linz"),
    target_sampling_rate_hz=200.0,
    bandpass_low_hz=4.5,
    bandpass_high_hz=5.5,
    filter_order=4,
    maximum_lag_s=2.0,
    detailed_window_start_s=10.0,
    detailed_window_end_s=20.0,
)


LOGGER = logging.getLogger("validate_GX010262_original_signals")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        output_dir / "validation.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(data),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


def prepare_uniform_signal(
    time: np.ndarray,
    values: np.ndarray,
    fs: float,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)

    if time.size != values.size:
        raise ValueError("Time and value arrays have different lengths.")

    valid = np.isfinite(time) & np.isfinite(values)
    time = time[valid]
    values = values[valid]

    if time.size < 2:
        raise ValueError("Too few valid samples.")

    order = np.argsort(time, kind="stable")
    time = time[order]
    values = values[order]

    unique_time, indices = np.unique(time, return_index=True)
    time = unique_time
    values = values[indices]

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis must be strictly increasing.")

    start_time = max(start_time, float(time[0]))
    end_time = min(end_time, float(time[-1]))

    if end_time <= start_time:
        raise ValueError("No valid interpolation interval.")

    uniform_time = np.arange(
        start_time,
        end_time,
        1.0 / fs,
        dtype=float,
    )

    if uniform_time.size < 2:
        raise ValueError("Uniform time grid is too short.")

    uniform_values = np.interp(
        uniform_time,
        time,
        values,
    )

    return uniform_time, uniform_values


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    std = float(np.std(values))

    if not np.isfinite(std) or std <= np.finfo(float).eps:
        raise ValueError("Signal standard deviation is zero or invalid.")

    return (values - np.mean(values)) / std


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    nyquist = fs / 2.0

    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError("Invalid band-pass configuration.")

    b, a = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
    )

    return np.asarray(
        filtfilt(b, a, signal),
        dtype=float,
    )


def estimate_lag(
    reference: np.ndarray,
    target: np.ndarray,
    fs: float,
    max_lag_s: float,
) -> tuple[int, float, float]:
    reference_z = zscore(reference)
    target_z = zscore(target)

    correlation = correlate(
        target_z,
        reference_z,
        mode="full",
    )

    lags = np.arange(
        -len(reference_z) + 1,
        len(target_z),
    )

    max_lag_samples = int(round(max_lag_s * fs))
    valid = np.abs(lags) <= max_lag_samples

    correlation_valid = correlation[valid]
    lags_valid = lags[valid]

    best_index = int(
        np.argmax(np.abs(correlation_valid))
    )

    lag_samples = int(lags_valid[best_index])
    lag_s = float(lag_samples / fs)

    normalized_peak = float(
        correlation_valid[best_index]
        / min(len(reference_z), len(target_z))
    )

    return lag_samples, lag_s, normalized_peak


def align_by_lag(
    reference: np.ndarray,
    target: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float)
    target = np.asarray(target, dtype=float)

    if lag_samples > 0:
        reference_aligned = reference[:-lag_samples]
        target_aligned = target[lag_samples:]
    elif lag_samples < 0:
        shift = -lag_samples
        reference_aligned = reference[shift:]
        target_aligned = target[:-shift]
    else:
        reference_aligned = reference
        target_aligned = target

    n = min(len(reference_aligned), len(target_aligned))

    if n < 2:
        raise ValueError("Too few samples remain after alignment.")

    return (
        reference_aligned[:n],
        target_aligned[:n],
    )


def calculate_metrics(
    video_signal: np.ndarray,
    imu_signal: np.ndarray,
    fs: float,
    maximum_lag_s: float,
) -> dict[str, Any]:
    lag_samples, lag_s, cross_correlation_peak = estimate_lag(
        video_signal,
        imu_signal,
        fs,
        maximum_lag_s,
    )

    video_aligned, imu_aligned = align_by_lag(
        video_signal,
        imu_signal,
        lag_samples,
    )

    video_z = zscore(video_aligned)
    imu_z = zscore(imu_aligned)

    pearson_r = float(
        np.corrcoef(video_z, imu_z)[0, 1]
    )

    rmse_zscore = float(
        np.sqrt(
            np.mean(
                (video_z - imu_z) ** 2
            )
        )
    )

    mae_zscore = float(
        np.mean(
            np.abs(video_z - imu_z)
        )
    )

    return {
        "lag_samples": lag_samples,
        "lag_s": lag_s,
        "cross_correlation_peak": cross_correlation_peak,
        "pearson_r": pearson_r,
        "abs_pearson_r": abs(pearson_r),
        "rmse_zscore": rmse_zscore,
        "mae_zscore": mae_zscore,
        "aligned_samples": int(len(video_aligned)),
        "aligned_duration_s": float(
            (len(video_aligned) - 1) / fs
        ),
    }


def load_video(config: ValidationConfig) -> pd.DataFrame:
    if not config.video_path.exists():
        raise FileNotFoundError(config.video_path)

    frame = pd.read_csv(config.video_path)

    required = {
        config.video_time_column,
        config.video_x_column,
        config.video_y_column,
    }

    missing = required.difference(frame.columns)

    if missing:
        raise KeyError(
            f"Missing required video columns: {sorted(missing)}"
        )

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame
        .dropna(subset=list(required))
        .sort_values(config.video_time_column)
        .reset_index(drop=True)
    )

    LOGGER.info("Video samples: %d", len(frame))

    return frame


def load_imu(config: ValidationConfig) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}

    for axis in config.imu_axes:
        result = load_imu_signal(
            config.imu_path,
            axis=axis,
        )

        output[axis] = {
            "time": np.asarray(result["time"], dtype=float),
            "signal": np.asarray(result["signal"], dtype=float),
        }

        LOGGER.info(
            "%s: %d samples, %.6f s, fs %.6f Hz",
            axis,
            int(result["num_samples"]),
            float(result["duration"]),
            float(result["sampling_rate"]),
        )

    return output


def plot_pair(
    output_path: Path,
    time: np.ndarray,
    video_signal: np.ndarray,
    imu_signal: np.ndarray,
    title: str,
    window_start: float | None = None,
    window_end: float | None = None,
) -> None:
    n = min(len(time), len(video_signal), len(imu_signal))

    time = time[:n]
    video_signal = zscore(video_signal[:n])
    imu_signal = zscore(imu_signal[:n])

    if window_start is not None and window_end is not None:
        mask = (
            (time >= window_start)
            & (time <= window_end)
        )
        time = time[mask]
        video_signal = video_signal[mask]
        imu_signal = imu_signal[mask]

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(
        time,
        video_signal,
        label="Video-derived displacement",
        linewidth=1.0,
    )

    ax.plot(
        time,
        imu_signal,
        label="IMU signal",
        linewidth=1.0,
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized amplitude [-]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_validation(config: ValidationConfig) -> pd.DataFrame:
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info("=" * 72)
    LOGGER.info("GX010262 original-signal external validation")
    LOGGER.info("=" * 72)

    video = load_video(config)
    imu_data = load_imu(config)

    video_time = video[
        config.video_time_column
    ].to_numpy(dtype=float)

    video_x = video[
        config.video_x_column
    ].to_numpy(dtype=float)

    video_y = video[
        config.video_y_column
    ].to_numpy(dtype=float)

    start_time = max(
        float(video_time[0]),
        *[
            float(data["time"][0])
            for data in imu_data.values()
        ],
    )

    end_time = min(
        float(video_time[-1]),
        *[
            float(data["time"][-1])
            for data in imu_data.values()
        ],
    )

    if end_time <= start_time:
        raise RuntimeError("Video and IMU do not overlap in time.")

    LOGGER.info(
        "Common interval: %.6f to %.6f s",
        start_time,
        end_time,
    )

    common_time, video_x_uniform = prepare_uniform_signal(
        video_time,
        video_x,
        config.target_sampling_rate_hz,
        start_time,
        end_time,
    )

    _, video_y_uniform = prepare_uniform_signal(
        video_time,
        video_y,
        config.target_sampling_rate_hz,
        start_time,
        end_time,
    )

    video_xy_uniform = np.sqrt(
        video_x_uniform**2
        + video_y_uniform**2
    )

    video_signals_raw = {
        "video_x_displacement_px": video_x_uniform,
        "video_y_displacement_px": video_y_uniform,
        "video_xy_displacement_px": video_xy_uniform,
    }

    imu_uniform: dict[str, np.ndarray] = {}

    for axis, data in imu_data.items():
        _, axis_uniform = prepare_uniform_signal(
            data["time"],
            data["signal"],
            config.target_sampling_rate_hz,
            start_time,
            end_time,
        )
        imu_uniform[axis] = axis_uniform

    imu_uniform["linxy_magnitude"] = np.sqrt(
        imu_uniform["linx"]**2
        + imu_uniform["liny"]**2
    )

    video_signals_band = {
        name: bandpass_filter(
            signal,
            config.target_sampling_rate_hz,
            config.bandpass_low_hz,
            config.bandpass_high_hz,
            config.filter_order,
        )
        for name, signal in video_signals_raw.items()
    }

    imu_signals_band = {
        name: bandpass_filter(
            signal,
            config.target_sampling_rate_hz,
            config.bandpass_low_hz,
            config.bandpass_high_hz,
            config.filter_order,
        )
        for name, signal in imu_uniform.items()
    }

    rows: list[dict[str, Any]] = []

    for video_name, video_raw in video_signals_raw.items():
        for imu_name, imu_raw in imu_uniform.items():
            LOGGER.info(
                "Evaluating %s vs %s",
                video_name,
                imu_name,
            )

            raw_metrics = calculate_metrics(
                video_raw,
                imu_raw,
                config.target_sampling_rate_hz,
                config.maximum_lag_s,
            )

            band_metrics = calculate_metrics(
                video_signals_band[video_name],
                imu_signals_band[imu_name],
                config.target_sampling_rate_hz,
                config.maximum_lag_s,
            )

            rows.append(
                {
                    "video_id": config.video_id,
                    "video_signal": video_name,
                    "imu_signal": imu_name,

                    "raw_lag_s": raw_metrics["lag_s"],
                    "raw_pearson_r": raw_metrics["pearson_r"],
                    "raw_abs_pearson_r": raw_metrics["abs_pearson_r"],
                    "raw_rmse_zscore": raw_metrics["rmse_zscore"],
                    "raw_mae_zscore": raw_metrics["mae_zscore"],

                    "band_lag_s": band_metrics["lag_s"],
                    "band_pearson_r": band_metrics["pearson_r"],
                    "band_abs_pearson_r": band_metrics["abs_pearson_r"],
                    "band_rmse_zscore": band_metrics["rmse_zscore"],
                    "band_mae_zscore": band_metrics["mae_zscore"],

                    "aligned_samples_band": band_metrics["aligned_samples"],
                    "aligned_duration_s_band": band_metrics["aligned_duration_s"],
                }
            )

    results = pd.DataFrame(rows)

    results = (
        results
        .sort_values(
            by=[
                "band_abs_pearson_r",
                "band_rmse_zscore",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    results.insert(
        0,
        "rank",
        np.arange(1, len(results) + 1),
    )

    results.to_csv(
        config.output_dir / "comparison.csv",
        index=False,
    )

    best = results.iloc[0].to_dict()

    best_video_name = str(best["video_signal"])
    best_imu_name = str(best["imu_signal"])

    raw_lag_samples = int(
        round(
            float(best["raw_lag_s"])
            * config.target_sampling_rate_hz
        )
    )

    band_lag_samples = int(
        round(
            float(best["band_lag_s"])
            * config.target_sampling_rate_hz
        )
    )

    raw_video_aligned, raw_imu_aligned = align_by_lag(
        video_signals_raw[best_video_name],
        imu_uniform[best_imu_name],
        raw_lag_samples,
    )

    band_video_aligned, band_imu_aligned = align_by_lag(
        video_signals_band[best_video_name],
        imu_signals_band[best_imu_name],
        band_lag_samples,
    )

    raw_time = (
        np.arange(len(raw_video_aligned), dtype=float)
        / config.target_sampling_rate_hz
    )

    band_time = (
        np.arange(len(band_video_aligned), dtype=float)
        / config.target_sampling_rate_hz
    )

    plot_pair(
        config.output_dir / "best_pair_raw_full_interval.png",
        raw_time,
        raw_video_aligned,
        raw_imu_aligned,
        (
            f"GX010262 original-signal comparison: "
            f"{best_video_name} vs {best_imu_name}"
        ),
    )

    plot_pair(
        config.output_dir / "best_pair_bandlimited_full_interval.png",
        band_time,
        band_video_aligned,
        band_imu_aligned,
        (
            f"GX010262 band-limited comparison "
            f"({config.bandpass_low_hz:.1f}-"
            f"{config.bandpass_high_hz:.1f} Hz)"
        ),
    )

    plot_pair(
        config.output_dir / "best_pair_bandlimited_10_20s.png",
        band_time,
        band_video_aligned,
        band_imu_aligned,
        (
            f"GX010262 band-limited comparison "
            f"({config.detailed_window_start_s:.0f}-"
            f"{config.detailed_window_end_s:.0f} s)"
        ),
        window_start=config.detailed_window_start_s,
        window_end=config.detailed_window_end_s,
    )

    config_data = asdict(config)
    config_data["created_utc"] = utc_now_iso()
    config_data["method_description"] = {
        "video_signal": (
            "Original Lucas-Kanade displacement; no numerical "
            "differentiation is applied."
        ),
        "video_xy": "sqrt(video_x^2 + video_y^2)",
        "imu_xy": "sqrt(linx^2 + liny^2)",
        "interpretation": (
            "Normalized signal-shape comparison only. "
            "Video displacement and IMU acceleration have different units."
        ),
    }

    write_json(
        config.output_dir / "validation_config.json",
        config_data,
    )

    write_json(
        config.output_dir / "summary.json",
        {
            "created_utc": utc_now_iso(),
            "video_id": config.video_id,
            "common_interval": {
                "start_time_s": float(start_time),
                "end_time_s": float(end_time),
                "duration_s": float(end_time - start_time),
                "uniform_samples": int(len(common_time)),
            },
            "number_of_comparisons": int(len(results)),
            "best_pair": to_jsonable(best),
        },
    )

    print()
    print("=" * 125)
    print("GX010262 ORIGINAL-SIGNAL VALIDATION RESULTS")
    print("=" * 125)

    print(
        results[
            [
                "rank",
                "video_signal",
                "imu_signal",
                "raw_lag_s",
                "raw_pearson_r",
                "band_lag_s",
                "band_pearson_r",
                "band_abs_pearson_r",
                "band_rmse_zscore",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("BEST PAIR")
    print("-" * 125)

    for key in [
        "video_signal",
        "imu_signal",
        "raw_lag_s",
        "raw_pearson_r",
        "band_lag_s",
        "band_pearson_r",
        "band_abs_pearson_r",
        "band_rmse_zscore",
    ]:
        print(f"{key}: {best[key]}")

    print()
    print(f"Output directory: {config.output_dir}")

    return results


def main() -> int:
    try:
        configure_logging(CONFIG.output_dir)
        run_validation(CONFIG)

        LOGGER.info(
            "Original-signal validation completed successfully."
        )
        return 0

    except Exception:
        LOGGER.exception(
            "GX010262 original-signal validation failed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())