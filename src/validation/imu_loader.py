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

    # ----- time axis -----

    # Preferred format of the experimental IMU data:
    # separate calendar date and clock-time columns, e.g.
    #
    # day         time
    # 2022-04-16  19:46:31.122418
    #
    # Convert the absolute timestamps to a relative time axis
    # starting at t = 0 s. This preserves the actual non-uniform
    # sample timing contained in the measurement file.

    if "day" in df.columns and "time" in df.columns:
        timestamps = pd.to_datetime(
            df["day"].astype(str)
            + " "
            + df["time"].astype(str),
            errors="coerce",
        )

        valid_timestamps = timestamps.notna()

        if valid_timestamps.sum() < 2:
            raise IMULoaderError(
                "Could not construct a valid time axis from "
                "'day' and 'time'."
            )

        # Keep invalid timestamps as NaN so that they can be removed
        # together with invalid signal values below.
        time = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        first_timestamp = timestamps.loc[
            valid_timestamps
        ].iloc[0]

        time[valid_timestamps.to_numpy()] = (
            timestamps.loc[valid_timestamps]
            - first_timestamp
        ).dt.total_seconds().to_numpy(
            dtype=float
        )

    else:
        possible_time_columns = [
            "time_seconds",
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
            time = pd.to_numeric(
                df[time_column],
                errors="coerce",
            ).to_numpy(
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