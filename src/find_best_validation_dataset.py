#!/usr/bin/env python3
"""Rank Lucas–Kanade video time series against IMU data.csv measurements."""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import detrend, periodogram

VIDEO_PATTERN = "*results_inner_pipe_track_timeseries.csv"


@dataclass
class Features:
    source_type: str
    identifier: str
    path: str
    signal_column: str
    time_column: str
    samples: int
    duration_s: float
    sampling_rate_hz: float
    dominant_frequency_hz: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    rms: float
    peak_to_peak: float
    quality_score: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find promising Video–IMU validation pairs.")
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--imu-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--video-column", default="inner_pipe_track_center_x")
    p.add_argument("--imu-column", default="liny")
    p.add_argument("--video-time-column", default=None)
    p.add_argument("--min-frequency", type=float, default=0.05)
    p.add_argument("--max-frequency", type=float, default=20.0)
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-imu-files", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts = [
        dict(sep=None, engine="python"),
        dict(sep=r"\s+", engine="python"),
        dict(sep=";", engine="python"),
        dict(sep=",", engine="python"),
    ]
    last = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as exc:
            last = exc
    raise ValueError(f"CSV could not be parsed: {last}")


def numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' missing. Available: {', '.join(df.columns)}")
    return pd.to_numeric(df[column], errors="coerce").to_numpy(float)


def detect_video_signal(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    candidates = [c for c in df.columns if "center_x" in c.lower() or "displacement" in c.lower()]
    if candidates:
        return candidates[0]
    excluded = {"frame", "frame_index", "index", "time", "time_s", "timestamp", "seconds", "t"}
    candidates = [c for c in df.columns if c.lower() not in excluded and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 10]
    if not candidates:
        raise KeyError("No usable numeric video signal column found.")
    return candidates[0]


def detect_video_time(df: pd.DataFrame, explicit: Optional[str]) -> tuple[np.ndarray, str]:
    names = ([explicit] if explicit else []) + [
    "time_seconds",
    "time_s",
    "time",
    "timestamp_s",
    "timestamp",
    "seconds",
    "t",
    "elapsed_time_s",
]
    for col in names:
        if col and col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if finite.size >= 3 and np.nanmedian(np.diff(finite)) > 0:
                return values, col
    frame_col = next((c for c in ["frame", "frame_index", "frame_idx"] if c in df.columns), None)
    fps_col = next((c for c in ["fps", "video_fps", "frame_rate"] if c in df.columns), None)
    if frame_col and fps_col:
        frames = numeric(df, frame_col)
        fps = float(np.nanmedian(numeric(df, fps_col)))
        if np.isfinite(fps) and fps > 0:
            return frames / fps, f"{frame_col}/{fps_col}"
    raise KeyError("No video time column found. Use --video-time-column if necessary.")


def detect_imu_time(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "day" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["day"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(), errors="coerce")
        valid = dt.notna()
        if valid.sum() >= 3:
            return (dt - dt[valid].iloc[0]).dt.total_seconds().to_numpy(float), "day+time"
    for col in ["timestamp", "datetime", "time_s", "seconds", "t"]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 3:
            return values.to_numpy(float), col
        dt = pd.to_datetime(df[col], errors="coerce")
        valid = dt.notna()
        if valid.sum() >= 3:
            return (dt - dt[valid].iloc[0]).dt.total_seconds().to_numpy(float), col
    raise KeyError("No usable IMU time information found.")


def prepare(time_s: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mask = np.isfinite(time_s) & np.isfinite(signal)
    t, x = np.asarray(time_s[mask], float), np.asarray(signal[mask], float)
    if len(t) < 16:
        raise ValueError("Fewer than 16 valid samples.")
    order = np.argsort(t)
    t, x = t[order], x[order]
    t, unique_idx = np.unique(t, return_index=True)
    x = x[unique_idx]
    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0]
    if positive_dt.size < 3:
        raise ValueError("Invalid time axis.")
    fs = 1.0 / float(np.median(positive_dt))
    duration = float(t[-1] - t[0])
    n = max(16, int(round(duration * fs)) + 1)
    tu = np.linspace(t[0], t[-1], n)
    xu = np.interp(tu, t, x)
    return tu, xu, fs


def compute_features(time_s, signal, source_type, identifier, path, signal_column, time_column, args) -> Features:
    t, x, fs = prepare(time_s, signal)
    duration = float(t[-1] - t[0])
    if duration < args.min_duration:
        raise ValueError(f"Duration {duration:.3f}s is too short.")
    x = detrend(x, type="linear")
    rms = float(np.sqrt(np.mean(x**2)))
    if rms <= np.finfo(float).eps:
        raise ValueError("Signal is effectively constant.")
    f, p = periodogram(x, fs=fs, window="hann", scaling="spectrum", detrend=False)
    upper = min(args.max_frequency, 0.98 * fs / 2.0)
    band = np.isfinite(f) & np.isfinite(p) & (f >= args.min_frequency) & (f <= upper)
    fb, pb = f[band], p[band]
    if fb.size < 3 or float(np.sum(pb)) <= 0:
        raise ValueError("Insufficient spectral content.")
    i = int(np.argmax(pb))
    total = float(np.sum(pb))
    centroid = float(np.sum(fb * pb) / total)
    bandwidth = float(np.sqrt(np.sum(((fb - centroid) ** 2) * pb) / total))
    prominence = float(pb[i] / (np.median(pb) + np.finfo(float).eps))
    quality = 100.0 * (0.35 * min(1.0, duration / 20.0) + 0.25 * min(1.0, fs / max(2.5 * upper, 1.0)) + 0.40 * min(1.0, math.log10(max(prominence, 1.0)) / 3.0))
    return Features(source_type, identifier, str(path), signal_column, time_column, len(x), duration, fs, float(fb[i]), centroid, bandwidth, rms, float(np.ptp(x)), float(np.clip(quality, 0, 100)))


def failed(source_type, identifier, path, column, exc) -> Features:
    nan = float("nan")
    return Features(source_type, identifier, str(path), column, "", 0, nan, nan, nan, nan, nan, nan, nan, 0.0, f"{type(exc).__name__}: {exc}")


def video_id(path: Path) -> str:
    m = re.search(r"(GX\d+)", path.name, re.I)
    return m.group(1).upper() if m else path.stem


def measurement_id(path: Path) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}", path.parent.name)
    return m.group(0) if m else path.parent.name


def analyse_video(path: Path, args) -> Features:
    ident = video_id(path)
    try:
        df = read_csv_flexible(path)
        col = detect_video_signal(df, args.video_column)
        t, tcol = detect_video_time(df, args.video_time_column)
        return compute_features(t, numeric(df, col), "video", ident, path, col, tcol, args)
    except Exception as exc:
        return failed("video", ident, path, args.video_column, exc)


def analyse_imu(path: Path, args) -> Features:
    ident = measurement_id(path)
    try:
        df = read_csv_flexible(path)
        t, tcol = detect_imu_time(df)
        return compute_features(t, numeric(df, args.imu_column), "imu", ident, path, args.imu_column, tcol, args)
    except Exception as exc:
        return failed("imu", ident, path, args.imu_column, exc)


def rel_diff(a: float, b: float, floor: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def rank_pairs(videos: pd.DataFrame, imus: pd.DataFrame, top_k: int) -> pd.DataFrame:
    videos = videos[videos["error"].fillna("") == ""]
    imus = imus[imus["error"].fillna("") == ""]
    rows = []
    for _, v in videos.iterrows():
        local = []
        for _, i in imus.iterrows():
            f_rel = rel_diff(v.dominant_frequency_hz, i.dominant_frequency_hz, 0.1)
            c_rel = rel_diff(v.spectral_centroid_hz, i.spectral_centroid_hz, 0.1)
            b_rel = rel_diff(v.spectral_bandwidth_hz, i.spectral_bandwidth_hz, 0.1)
            d_rel = rel_diff(v.duration_s, i.duration_s, 1.0)
            quality = min(v.quality_score, i.quality_score) / 100.0
            score = 100.0 * (0.55 * math.exp(-4*f_rel) + 0.15 * math.exp(-2*c_rel) + 0.10 * math.exp(-1.5*b_rel) + 0.10 * math.exp(-1.5*d_rel) + 0.10 * quality)
            local.append({
                "video_id": v.identifier,
                "video_path": v.path,
                "imu_measurement_id": i.identifier,
                "imu_path": i.path,
                "video_dominant_frequency_hz": v.dominant_frequency_hz,
                "imu_dominant_frequency_hz": i.dominant_frequency_hz,
                "frequency_error_hz": abs(v.dominant_frequency_hz - i.dominant_frequency_hz),
                "frequency_error_percent": 100.0 * f_rel,
                "video_duration_s": v.duration_s,
                "imu_duration_s": i.duration_s,
                "video_quality_score": v.quality_score,
                "imu_quality_score": i.quality_score,
                "score": score,
            })
        local.sort(key=lambda r: r["score"], reverse=True)
        for rank, row in enumerate(local[:top_k], 1):
            row["rank_for_video"] = rank
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    video_root = args.video_root.expanduser().resolve()
    imu_root = args.imu_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not video_root.is_dir():
        print(f"ERROR: video directory not found: {video_root}", file=sys.stderr)
        return 2
    if not imu_root.is_dir():
        print(f"ERROR: IMU directory not found: {imu_root}", file=sys.stderr)
        return 2
    videos = sorted(video_root.rglob(VIDEO_PATTERN))
    imus = sorted(imu_root.rglob("data.csv"))
    if args.max_imu_files:
        imus = imus[:args.max_imu_files]
    if not videos:
        print(f"ERROR: no files matching {VIDEO_PATTERN} below {video_root}", file=sys.stderr)
        return 2
    if not imus:
        print(f"ERROR: no data.csv below {imu_root}", file=sys.stderr)
        return 2

    vf_path, imf_path = out / "video_features.csv", out / "imu_features.csv"
    if vf_path.exists() and not args.force:
        vf = pd.read_csv(vf_path)
    else:
        items = []
        for n, path in enumerate(videos, 1):
            print(f"[Video {n}/{len(videos)}] {path.name}")
            items.append(asdict(analyse_video(path, args)))
        vf = pd.DataFrame(items)
        vf.to_csv(vf_path, index=False)

    if imf_path.exists() and not args.force:
        imf = pd.read_csv(imf_path)
    else:
        items = []
        for n, path in enumerate(imus, 1):
            print(f"[IMU {n}/{len(imus)}] {path}")
            items.append(asdict(analyse_imu(path, args)))
        imf = pd.DataFrame(items)
        imf.to_csv(imf_path, index=False)

    ranked = rank_pairs(vf, imf, args.top_k)
    ranked.to_csv(out / "validation_candidates_ranked.csv", index=False)
    if ranked.empty:
        best = ranked.copy()
    else:
        best = ranked.sort_values("score", ascending=False).groupby("video_id", as_index=False).first().sort_values("score", ascending=False)
    best.to_csv(out / "best_candidate_per_video.csv", index=False)
    vf[vf.error.fillna("") != ""].to_csv(out / "failed_video_files.csv", index=False)
    imf[imf.error.fillna("") != ""].to_csv(out / "failed_imu_files.csv", index=False)

    print("\nDone")
    print(f"Videos: {len(vf)}; valid: {(vf.error.fillna('') == '').sum()}")
    print(f"IMUs:   {len(imf)}; valid: {(imf.error.fillna('') == '').sum()}")
    print(f"Output: {out}")
    if not best.empty:
        cols = ["video_id", "imu_measurement_id", "video_dominant_frequency_hz", "imu_dominant_frequency_hz", "frequency_error_hz", "score"]
        print("\nBest candidates:")
        print(best[cols].head(20).to_string(index=False))
    print("\nNote: A high score is a shortlist criterion, not proof of simultaneous recording. Check logs/date/experiment conditions before final selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())    