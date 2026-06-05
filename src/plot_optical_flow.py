import os
import pandas as pd
import matplotlib.pyplot as plt

CSV = "outputs/optical_flow/optical_flow_points.csv"
OUT_DIR = "outputs/optical_flow"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV)

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
plt.savefig(os.path.join(OUT_DIR, "optical_flow_y.png"))
plt.close()

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
plt.savefig(os.path.join(OUT_DIR, "optical_flow_dy.png"))
plt.close()

print("Plots gespeichert in:", OUT_DIR)