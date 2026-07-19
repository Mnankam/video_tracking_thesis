"""
video_loader.py

Load a video tracking time series exported by the tracking pipeline.

Author: Serge Kouomnankam
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SIGNAL = "inner_pipe_track_center_x"


class VideoLoaderError(Exception):
    """Raised when the video file cannot be loaded."""


def load_video_signal(
    csv_file,
    signal_column=DEFAULT_SIGNAL,
    valid_only=True,
):
    """
    Load a video tracking signal.

    Parameters
    ----------
    csv_file : str | Path
        Path to *_track_timeseries.csv

    signal_column : str
        Signal column to use.

    valid_only : bool
        Use only rows with inner_pipe_track_valid == 1.

    Returns
    -------
    dict
    """

    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)

    required = [
        "time_seconds",
        signal_column,
        "inner_pipe_track_valid",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise VideoLoaderError(
            f"Missing columns: {missing}"
        )

    if valid_only:
        df = df[df["inner_pipe_track_valid"] == 1].copy()

    df = df.sort_values("time_seconds")

    time = df["time_seconds"].to_numpy(dtype=float)

    signal = df[signal_column].to_numpy(dtype=float)

    mask = np.isfinite(time) & np.isfinite(signal)

    time = time[mask]
    signal = signal[mask]

    if len(time) < 2:
        raise VideoLoaderError(
            "Not enough valid samples."
        )

    dt = np.diff(time)

    sampling_rate = float(1.0 / np.median(dt))

    duration = float(time[-1] - time[0])

    return {
        "time": time,
        "signal": signal,
        "sampling_rate": sampling_rate,
        "duration": duration,
        "num_samples": len(signal),
        "signal_name": signal_column,
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("csv")

    parser.add_argument(
        "--signal",
        default=DEFAULT_SIGNAL,
    )

    args = parser.parse_args()

    data = load_video_signal(
        args.csv,
        signal_column=args.signal,
    )

    print("Video loaded")
    print("----------------------------")
    print(f"Samples       : {data['num_samples']}")
    print(f"Duration [s]  : {data['duration']:.3f}")
    print(f"Sampling Rate : {data['sampling_rate']:.2f} Hz")
    print(f"Signal        : {data['signal_name']}")