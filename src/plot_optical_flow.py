import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Optical Flow results")
    parser.add_argument("--csv", required=True, help="Optical-Flow-CSV-Datei")
    parser.add_argument("--out-dir", required=True, help="Ausgabeordner für Plots")
    args = parser.parse_args()

    csv_path = args.csv
    out_dir = args.out_dir

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if df.empty:
        raise RuntimeError(f"CSV ist leer: {csv_path}")

    # Nur gültige Trackingpunkte verwenden
    if "tracking_status" in df.columns:
        df = df[df["tracking_status"] == 1].copy()

    # =========================================================
    # y position
    # =========================================================

    plt.figure(figsize=(10, 5))

    for pid in sorted(df["point_id"].unique()):
        d = df[df["point_id"] == pid]
        plt.plot(d["frame"], d["y"], label=f"point {pid}")

    plt.xlabel("Frame")
    plt.ylabel("y position [px]")
    plt.title("Optical Flow: y-position over time")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "optical_flow_y.png"),
        dpi=150,
    )

    plt.close()

    # =========================================================
    # dy
    # =========================================================

    plt.figure(figsize=(10, 5))

    for pid in sorted(df["point_id"].unique()):
        d = df[df["point_id"] == pid]
        plt.plot(d["frame"], d["dy"], label=f"point {pid}")

    plt.xlabel("Frame")
    plt.ylabel("dy [px/frame]")
    plt.title("Optical Flow: vertical displacement")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "optical_flow_dy.png"),
        dpi=150,
    )

    plt.close()

    # =========================================================
    # x position
    # =========================================================

    plt.figure(figsize=(10, 5))

    for pid in sorted(df["point_id"].unique()):
        d = df[df["point_id"] == pid]
        plt.plot(d["frame"], d["x"], label=f"point {pid}")

    plt.xlabel("Frame")
    plt.ylabel("x position [px]")
    plt.title("Optical Flow: x-position over time")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "optical_flow_x.png"),
        dpi=150,
    )

    plt.close()

    # =========================================================
    # dx
    # =========================================================

    plt.figure(figsize=(10, 5))

    for pid in sorted(df["point_id"].unique()):
        d = df[df["point_id"] == pid]
        plt.plot(d["frame"], d["dx"], label=f"point {pid}")

    plt.xlabel("Frame")
    plty.label("dx [px/frame]")
    plt.title("Optical Flow: horizontal displacement")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "optical_flow_dx.png"),
        dpi=150,
    )

    plt.close()

    # =========================================================
    # displacement magnitude
    # =========================================================

    plt.figure(figsize=(10, 5))

    for pid in sorted(df["point_id"].unique()):

        d = df[df["point_id"] == pid].copy()

        disp = (d["dx"] ** 2 + d["dy"] ** 2) ** 0.5

        plt.plot(
            d["frame"],
            disp,
            label=f"point {pid}",
        )

    plt.xlabel("Frame")
    plt.ylabel("displacement magnitude [px/frame]")

    plt.title("Optical Flow: displacement magnitude")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "optical_flow_displacement.png"),
        dpi=150,
    )

    plt.close()

    print(f"Plots gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()