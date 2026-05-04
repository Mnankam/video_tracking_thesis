import pandas as pd
import matplotlib.pyplot as plt

file = "/scratch-scc/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/batch/batch_comparison.csv"
df = pd.read_csv(file)

# Sortieren
df = df.sort_values(by="num_detections", ascending=False)

# Top 15 anzeigen
df_top = df.head(15)

plt.figure()
plt.bar(df_top["video"], df_top["num_detections"])
plt.xticks(rotation=90)
plt.title("Top Videos by Number of Detections")
plt.xlabel("Video")
plt.ylabel("Detections")
plt.tight_layout()
plt.savefig("/scratch-scc/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/batch/detections_plot.png")

# Tracks Plot
plt.figure()
plt.bar(df_top["video"], df_top["num_tracks"])
plt.xticks(rotation=90)
plt.title("Top Videos by Number of Tracks")
plt.xlabel("Video")
plt.ylabel("Tracks")
plt.tight_layout()
plt.savefig("/scratch-scc/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/batch/tracks_plot.png")

print("Plots saved!")
