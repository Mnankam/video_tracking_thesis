import pandas as pd
from pathlib import Path

BATCH_DIR = Path("/scratch-scc/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/batch")
OUT_FILE = BATCH_DIR / "batch_comparison.csv"

rows = []

for result_file in sorted(BATCH_DIR.glob("*_results.csv")):
    video = result_file.name.replace("_results.csv", "")

    df = pd.read_csv(result_file)

    rows.append({
        "video": video,
        "num_detections": len(df),
        "num_frames": df["frame"].nunique() if "frame" in df.columns else None,
        "num_tracks": df["track_id"].nunique() if "track_id" in df.columns else None,
        "mean_score": df["score"].mean() if "score" in df.columns else None,
        "mean_area": df["area"].mean() if "area" in df.columns else None,
    })

comparison = pd.DataFrame(rows)
comparison = comparison.sort_values(by="num_detections", ascending=False)
comparison.to_csv(OUT_FILE, index=False)

print(comparison)
print(f"\nSaved to: {OUT_FILE}")
