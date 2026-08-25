#!/usr/bin/env python3
"""
validate_GX010262_original_coherence.py

Frequency-domain external validation for GX010262 using the original
Lucas-Kanade displacement signals. No numerical differentiation is applied.

Video signals:
- video_x_displacement_px
- video_y_displacement_px
- video_xy_displacement_px = sqrt(x^2 + y^2)

IMU signals:
- linx
- liny
- linz
- linxy_magnitude = sqrt(linx^2 + liny^2)

The script evaluates Welch PSD and magnitude-squared coherence for all
12 video/IMU pairs.

Coherence measures frequency-dependent linear coupling. It does not imply
absolute amplitude agreement, identical physical units, or causality.
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
from scipy.signal import coherence, find_peaks, welch

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
    analysis_low_hz: float
    analysis_high_hz: float
    coherence_band_low_hz: float
    coherence_band_high_hz: float
    reference_frequency_hz: float
    reference_frequency_tolerance_hz: float
    welch_nperseg: int
    welch_overlap_fraction: float


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
        "GX010262_original_coherence"
    ),
    video_time_column="time_seconds",
    video_x_column="lk_displacement_x",
    video_y_column="lk_displacement_y",
    imu_axes=("linx", "liny", "linz"),
    target_sampling_rate_hz=200.0,
    analysis_low_hz=2.0,
    analysis_high_hz=8.0,
    coherence_band_low_hz=4.5,
    coherence_band_high_hz=5.5,
    reference_frequency_hz=5.0,
    reference_frequency_tolerance_hz=0.15,
    welch_nperseg=2048,
    welch_overlap_fraction=0.5,
)


LOGGER = logging.getLogger("validate_GX010262_original_coherence")


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
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
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

    unique_time, indices = np.unique(
        time,
        return_index=True,
    )

    time = unique_time
    values = values[indices]

    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis must be strictly increasing.")

    start_time = max(
        start_time,
        float(time[0]),
    )

    end_time = min(
        end_time,
        float(time[-1]),
    )

    if end_time <= start_time:
        raise ValueError("No valid common interpolation interval.")

    uniform_time = np.arange(
        start_time,
        end_time,
        1.0 / fs,
        dtype=float,
    )

    if uniform_time.size < 2:
        raise ValueError("Uniform time axis is too short.")

    uniform_values = np.interp(
        uniform_time,
        time,
        values,
    )

    return uniform_time, uniform_values


def remove_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - np.mean(values)


def get_welch_parameters(
    signal_length: int,
    configured_nperseg: int,
    overlap_fraction: float,
) -> tuple[int, int]:
    if signal_length < 8:
        raise ValueError("Signal is too short for Welch analysis.")

    nperseg = min(
        configured_nperseg,
        signal_length,
    )

    noverlap = int(
        round(
            nperseg
            * overlap_fraction
        )
    )

    noverlap = min(
        noverlap,
        nperseg - 1,
    )

    return nperseg, noverlap


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, psd = welch(
        remove_mean(signal),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    return frequencies, psd


def compute_coherence_curve(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, values = coherence(
        remove_mean(signal_a),
        remove_mean(signal_b),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )

    return frequencies, values


def nearest_frequency_value(
    frequencies: np.ndarray,
    values: np.ndarray,
    target_frequency_hz: float,
) -> tuple[float, float]:
    index = int(
        np.argmin(
            np.abs(
                frequencies
                - target_frequency_hz
            )
        )
    )

    return (
        float(frequencies[index]),
        float(values[index]),
    )


def band_statistics(
    frequencies: np.ndarray,
    values: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> dict[str, float]:
    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    if not np.any(mask):
        raise ValueError(
            f"No bins inside {low_hz}-{high_hz} Hz."
        )

    band_f = frequencies[mask]
    band_v = values[mask]
    max_index = int(np.argmax(band_v))

    return {
        "band_max_value": float(
            band_v[max_index]
        ),
        "band_max_frequency_hz": float(
            band_f[max_index]
        ),
        "band_mean_value": float(
            np.mean(band_v)
        ),
        "band_median_value": float(
            np.median(band_v)
        ),
    }


def dominant_frequency_in_band(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> tuple[float, float]:
    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    if not np.any(mask):
        raise ValueError("No frequencies in requested band.")

    band_f = frequencies[mask]
    band_psd = psd[mask]

    index = int(np.argmax(band_psd))

    return (
        float(band_f[index]),
        float(band_psd[index]),
    )


def top_spectral_peaks(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
    limit: int = 10,
) -> list[dict[str, float]]:
    mask = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    local_f = frequencies[mask]
    local_psd = psd[mask]

    if local_f.size < 3:
        return []

    peaks, _ = find_peaks(local_psd)

    if peaks.size == 0:
        return []

    order = peaks[
        np.argsort(
            local_psd[peaks]
        )[::-1]
    ]

    return [
        {
            "frequency_hz": float(local_f[i]),
            "psd": float(local_psd[i]),
        }
        for i in order[:limit]
    ]


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
            f"Missing video columns: {sorted(missing)}"
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


def load_imu(
    config: ValidationConfig,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}

    for axis in config.imu_axes:
        result = load_imu_signal(
            config.imu_path,
            axis=axis,
        )

        output[axis] = {
            "time": np.asarray(
                result["time"],
                dtype=float,
            ),
            "signal": np.asarray(
                result["signal"],
                dtype=float,
            ),
        }

        LOGGER.info(
            "%s: %d samples, %.6f s, fs %.6f Hz",
            axis,
            int(result["num_samples"]),
            float(result["duration"]),
            float(result["sampling_rate"]),
        )

    return output


def create_pair_plot(
    output_path: Path,
    video_name: str,
    imu_name: str,
    frequencies: np.ndarray,
    coherence_values: np.ndarray,
    video_psd_f: np.ndarray,
    video_psd: np.ndarray,
    imu_psd_f: np.ndarray,
    imu_psd: np.ndarray,
    config: ValidationConfig,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        frequencies,
        coherence_values,
    )

    ax.axvline(
        config.reference_frequency_hz,
        linestyle="--",
        linewidth=1,
        label="5 Hz reference",
    )

    ax.axvspan(
        config.coherence_band_low_hz,
        config.coherence_band_high_hz,
        alpha=0.15,
        label="4.5-5.5 Hz band",
    )

    ax.set_xlim(
        config.analysis_low_hz,
        config.analysis_high_hz,
    )

    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude-squared coherence")

    ax.set_title(
        f"GX010262 coherence: {video_name} vs {imu_name}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    psd_path = (
        output_path.parent
        / f"{output_path.stem}_psd.png"
    )

    video_mask = (
        (video_psd_f >= config.analysis_low_hz)
        & (video_psd_f <= config.analysis_high_hz)
    )

    imu_mask = (
        (imu_psd_f >= config.analysis_low_hz)
        & (imu_psd_f <= config.analysis_high_hz)
    )

    video_values = video_psd[video_mask].copy()
    imu_values = imu_psd[imu_mask].copy()

    if np.max(video_values) > 0:
        video_values /= np.max(video_values)

    if np.max(imu_values) > 0:
        imu_values /= np.max(imu_values)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        video_psd_f[video_mask],
        video_values,
        label=video_name,
    )

    ax.plot(
        imu_psd_f[imu_mask],
        imu_values,
        label=imu_name,
    )

    ax.axvline(
        config.reference_frequency_hz,
        linestyle="--",
        linewidth=1,
        label="5 Hz reference",
    )

    ax.axvspan(
        config.coherence_band_low_hz,
        config.coherence_band_high_hz,
        alpha=0.15,
        label="4.5-5.5 Hz band",
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized PSD")

    ax.set_title(
        f"GX010262 PSD comparison: {video_name} vs {imu_name}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        psd_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def evaluate_pair(
    video_name: str,
    video_signal: np.ndarray,
    imu_name: str,
    imu_signal: np.ndarray,
    fs: float,
    config: ValidationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signal_length = min(
        len(video_signal),
        len(imu_signal),
    )

    video_signal = np.asarray(
        video_signal[:signal_length],
        dtype=float,
    )

    imu_signal = np.asarray(
        imu_signal[:signal_length],
        dtype=float,
    )

    nperseg, noverlap = get_welch_parameters(
        signal_length,
        config.welch_nperseg,
        config.welch_overlap_fraction,
    )

    coherence_f, coherence_values = compute_coherence_curve(
        video_signal,
        imu_signal,
        fs,
        nperseg,
        noverlap,
    )

    video_psd_f, video_psd = compute_psd(
        video_signal,
        fs,
        nperseg,
        noverlap,
    )

    imu_psd_f, imu_psd = compute_psd(
        imu_signal,
        fs,
        nperseg,
        noverlap,
    )

    band = band_statistics(
        coherence_f,
        coherence_values,
        config.coherence_band_low_hz,
        config.coherence_band_high_hz,
    )

    ref_bin_hz, coherence_at_ref = nearest_frequency_value(
        coherence_f,
        coherence_values,
        config.reference_frequency_hz,
    )

    reference_mask = (
        coherence_f
        >= (
            config.reference_frequency_hz
            - config.reference_frequency_tolerance_hz
        )
    ) & (
        coherence_f
        <= (
            config.reference_frequency_hz
            + config.reference_frequency_tolerance_hz
        )
    )

    if np.any(reference_mask):
        coherence_near_ref_mean = float(
            np.mean(
                coherence_values[reference_mask]
            )
        )

        coherence_near_ref_max = float(
            np.max(
                coherence_values[reference_mask]
            )
        )
    else:
        coherence_near_ref_mean = float("nan")
        coherence_near_ref_max = float("nan")

    video_dom_f, video_dom_psd = dominant_frequency_in_band(
        video_psd_f,
        video_psd,
        config.analysis_low_hz,
        config.analysis_high_hz,
    )

    imu_dom_f, imu_dom_psd = dominant_frequency_in_band(
        imu_psd_f,
        imu_psd,
        config.analysis_low_hz,
        config.analysis_high_hz,
    )

    metrics = {
        "video_signal": video_name,
        "imu_signal": imu_name,
        "samples": int(signal_length),
        "duration_s": float(
            (signal_length - 1) / fs
        ),
        "sampling_rate_hz": float(fs),
        "welch_nperseg": int(nperseg),
        "welch_noverlap": int(noverlap),
        "frequency_resolution_hz": float(
            coherence_f[1] - coherence_f[0]
        ),
        "coherence_at_5hz": float(coherence_at_ref),
        "coherence_5hz_bin_hz": float(ref_bin_hz),
        "coherence_near_5hz_mean": float(
            coherence_near_ref_mean
        ),
        "coherence_near_5hz_max": float(
            coherence_near_ref_max
        ),
        "band_max_coherence": float(
            band["band_max_value"]
        ),
        "band_max_coherence_frequency_hz": float(
            band["band_max_frequency_hz"]
        ),
        "band_mean_coherence": float(
            band["band_mean_value"]
        ),
        "band_median_coherence": float(
            band["band_median_value"]
        ),
        "video_dominant_frequency_2_8_hz": float(
            video_dom_f
        ),
        "imu_dominant_frequency_2_8_hz": float(
            imu_dom_f
        ),
        "dominant_frequency_difference_hz": float(
            abs(video_dom_f - imu_dom_f)
        ),
        "video_distance_to_5hz": float(
            abs(
                video_dom_f
                - config.reference_frequency_hz
            )
        ),
        "imu_distance_to_5hz": float(
            abs(
                imu_dom_f
                - config.reference_frequency_hz
            )
        ),
        "video_dominant_psd": float(video_dom_psd),
        "imu_dominant_psd": float(imu_dom_psd),
    }

    diagnostics = {
        "coherence_frequency": coherence_f,
        "coherence": coherence_values,
        "video_psd_frequency": video_psd_f,
        "video_psd": video_psd,
        "imu_psd_frequency": imu_psd_f,
        "imu_psd": imu_psd,
        "video_top_peaks": top_spectral_peaks(
            video_psd_f,
            video_psd,
            config.analysis_low_hz,
            config.analysis_high_hz,
            limit=10,
        ),
        "imu_top_peaks": top_spectral_peaks(
            imu_psd_f,
            imu_psd,
            config.analysis_low_hz,
            config.analysis_high_hz,
            limit=10,
        ),
    }

    return metrics, diagnostics


def run_validation(
    config: ValidationConfig,
) -> pd.DataFrame:
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = (
        config.output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info("=" * 72)
    LOGGER.info("GX010262 original-signal coherence validation")
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
        raise RuntimeError(
            "No temporal overlap between video and IMU."
        )

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

    video_signals = {
        "video_x_displacement_px": video_x_uniform,
        "video_y_displacement_px": video_y_uniform,
        "video_xy_displacement_px": video_xy_uniform,
    }

    imu_signals: dict[str, np.ndarray] = {}

    for axis, data in imu_data.items():
        _, axis_uniform = prepare_uniform_signal(
            data["time"],
            data["signal"],
            config.target_sampling_rate_hz,
            start_time,
            end_time,
        )

        imu_signals[axis] = axis_uniform

    imu_signals["linxy_magnitude"] = np.sqrt(
        imu_signals["linx"]**2
        + imu_signals["liny"]**2
    )

    rows: list[dict[str, Any]] = []
    diagnostic_summary: dict[str, Any] = {}

    for video_name, video_signal in video_signals.items():
        for imu_name, imu_signal in imu_signals.items():
            LOGGER.info(
                "Evaluating coherence: %s vs %s",
                video_name,
                imu_name,
            )

            metrics, diagnostics = evaluate_pair(
                video_name,
                video_signal,
                imu_name,
                imu_signal,
                config.target_sampling_rate_hz,
                config,
            )

            rows.append(
                {
                    "video_id": config.video_id,
                    **metrics,
                }
            )

            key = f"{video_name}__{imu_name}"

            diagnostic_summary[key] = {
                "video_top_peaks": diagnostics[
                    "video_top_peaks"
                ],
                "imu_top_peaks": diagnostics[
                    "imu_top_peaks"
                ],
            }

            create_pair_plot(
                output_path=(
                    plot_dir
                    / f"{video_name}_vs_{imu_name}_coherence.png"
                ),
                video_name=video_name,
                imu_name=imu_name,
                frequencies=diagnostics[
                    "coherence_frequency"
                ],
                coherence_values=diagnostics[
                    "coherence"
                ],
                video_psd_f=diagnostics[
                    "video_psd_frequency"
                ],
                video_psd=diagnostics[
                    "video_psd"
                ],
                imu_psd_f=diagnostics[
                    "imu_psd_frequency"
                ],
                imu_psd=diagnostics[
                    "imu_psd"
                ],
                config=config,
            )

    results = pd.DataFrame(rows)

    results = (
        results
        .sort_values(
            by=[
                "coherence_at_5hz",
                "band_mean_coherence",
                "band_max_coherence",
            ],
            ascending=[
                False,
                False,
                False,
            ],
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

    config_data = asdict(config)
    config_data["created_utc"] = utc_now_iso()
    config_data["method"] = {
        "video_signal": (
            "Original Lucas-Kanade displacement; no numerical "
            "differentiation is applied."
        ),
        "video_xy": "sqrt(video_x^2 + video_y^2)",
        "imu_xy": "sqrt(linx^2 + liny^2)",
        "coherence": (
            "Magnitude-squared coherence using scipy.signal.coherence."
        ),
        "psd": (
            "Welch power spectral density with Hann window."
        ),
        "resampling": (
            "Video and IMU interpolated to a common 200 Hz grid."
        ),
        "interpretation": (
            "Frequency-domain coupling only. "
            "Absolute amplitude agreement is not evaluated."
        ),
    }

    write_json(
        config.output_dir
        / "validation_config.json",
        config_data,
    )

    write_json(
        config.output_dir
        / "spectral_peaks.json",
        diagnostic_summary,
    )

    best = results.iloc[0].to_dict()

    write_json(
        config.output_dir
        / "summary.json",
        {
            "created_utc": utc_now_iso(),
            "video_id": config.video_id,
            "common_interval": {
                "start_time_s": float(start_time),
                "end_time_s": float(end_time),
                "duration_s": float(end_time - start_time),
                "samples": int(len(common_time)),
            },
            "number_of_comparisons": int(len(results)),
            "best_pair": to_jsonable(best),
            "interpretation_note": (
                "Coherence is computed from original video displacement "
                "without numerical differentiation."
            ),
        },
    )

    print()
    print("=" * 125)
    print("GX010262 ORIGINAL-SIGNAL COHERENCE RESULTS")
    print("=" * 125)

    display_columns = [
        "rank",
        "video_signal",
        "imu_signal",
        "coherence_at_5hz",
        "coherence_near_5hz_mean",
        "coherence_near_5hz_max",
        "band_max_coherence",
        "band_max_coherence_frequency_hz",
        "band_mean_coherence",
        "video_dominant_frequency_2_8_hz",
        "imu_dominant_frequency_2_8_hz",
        "dominant_frequency_difference_hz",
    ]

    print(
        results[
            display_columns
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
        "coherence_at_5hz",
        "coherence_5hz_bin_hz",
        "coherence_near_5hz_mean",
        "coherence_near_5hz_max",
        "band_max_coherence",
        "band_max_coherence_frequency_hz",
        "band_mean_coherence",
        "band_median_coherence",
        "video_dominant_frequency_2_8_hz",
        "imu_dominant_frequency_2_8_hz",
        "dominant_frequency_difference_hz",
        "frequency_resolution_hz",
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
            "Original-signal coherence validation completed successfully."
        )
        return 0

    except Exception:
        LOGGER.exception(
            "GX010262 original-signal coherence validation failed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())