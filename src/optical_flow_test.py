import cv2
import numpy as np
import pandas as pd
import os

VIDEO_PATH = "/home/serge_muriel/projects/video_tracking/data/video.MP4"
OUT_CSV = "outputs/optical_flow/optical_flow_points.csv"

START_FRAME = 0
END_FRAME = 500

points = np.array([
    [180, 365],
    [350, 365],
    [520, 365],
    [180, 406],
    [350, 406],
    [520, 406],
], dtype=np.float32).reshape(-1, 1, 2)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Video konnte nicht geöffnet werden.")

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_points = points.copy()

rows = []

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

frame_idx = START_FRAME + 1

while frame_idx < END_FRAME:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        **lk_params,
    )

    for i, (p0, p1, st) in enumerate(zip(prev_points, next_points, status)):
        if st[0] == 1:
            x0, y0 = p0.ravel()
            x1, y1 = p1.ravel()

            rows.append({
                "frame": frame_idx,
                "point_id": i,
                "x": x1,
                "y": y1,
                "dx": x1 - x0,
                "dy": y1 - y0,
            })

    prev_gray = gray.copy()
    prev_points = next_points.copy()
    frame_idx += 1

cap.release()

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"Gespeichert: {OUT_CSV}")
print(df.head())