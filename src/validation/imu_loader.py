"""
imu_loader.py

Load an archived IMU measurement.

Author: Serge Kouomnankam
"""

from pathlib import Path

import numpy as np
import pandas as pd


class IMULoaderError(Exception):
    """Raised when IMU data cannot be loaded."""


def load_imu_signal(
    csv_file,
    axis="liny",
):
    """
    Load one IMU axis.

    Parameters
    ----------
    csv_file : str | Path
        Path to the IMU CSV file.

    axis : str
        Example:
        linx
        liny
        linz
        rotx
        roty
        rotz

    Returns
    -------
    dict
    """

    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(
        csv_file,
        sep=r"\s+",
    )

    # ----- time column -----

    possible_time_columns = [
        "time_seconds",
        "time",
        "timestamp",
        "t",
    ]

    time_column = None

    for col in possible_time_columns:
        if col in df.columns:
            time_column = col
            break

    if time_column is None:
        time = np.arange(
            len(df),
            dtype=float,
        )
    else:
        time = df[time_column].to_numpy(
            dtype=float
        )

    # ----- axis -----

    if axis not in df.columns:
        raise IMULoaderError(
            f"Axis '{axis}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    signal = df[axis].to_numpy(
        dtype=float
    )

    mask = (
        np.isfinite(time)
        & np.isfinite(signal)
    )

    time = time[mask]
    signal = signal[mask]

    if len(time) < 2:
        raise IMULoaderError(
            "Not enough valid samples."
        )

    dt = np.diff(time)

    if np.all(dt > 0):
        sampling_rate = float(
            1.0 / np.median(dt)
        )
    else:
        sampling_rate = np.nan

    duration = float(
        time[-1] - time[0]
    )

    return {
        "time": time,
        "signal": signal,
        "sampling_rate": sampling_rate,
        "duration": duration,
        "num_samples": len(signal),
        "signal_name": axis,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv"
    )

    parser.add_argument(
        "--axis",
        default="liny",
    )

    args = parser.parse_args()

    data = load_imu_signal(
        args.csv,
        axis=args.axis,
    )

    print("IMU loaded")
    print("----------------------------")
    print(
        f"Samples       : "
        f"{data['num_samples']}"
    )
    print(
        f"Duration [s]  : "
        f"{data['duration']:.3f}"
    )
    print(
        f"Sampling Rate : "
        f"{data['sampling_rate']:.2f} Hz"
    )
    print(
        f"Axis          : "
        f"{data['signal_name']}"
    )