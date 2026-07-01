import cv2
import os

video_path = "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/video_side/GX010044.MP4"
out_dir = "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/iou_gx010044/frames"

os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Video konnte nicht geöffnet werden.")

frames_to_save = list(range(0, 186, 10))  # 0,10,20,...,180

for frame_id in frames_to_save:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()

    if not ok:
        print("Could not read frame:", frame_id)
        continue

    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    out_path = os.path.join(out_dir, f"frame_{frame_id:06d}.png")
    cv2.imwrite(out_path, frame)
    print("saved:", out_path)

cap.release()
print("Done.")
