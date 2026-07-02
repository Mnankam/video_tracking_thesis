import os
import matplotlib.pyplot as plt

OUT_DIR = "thesis_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 1) Performance / FPS Plot
# =========================================================

methods = [
    "OpenCV\nPipeline",
    "Lucas-\nKanade",
    "Farneback\nDense",
    "Detectron2\nMask R-CNN",
]

fps = [
    76.00,
    47.64,
    11.98,
    25.64,
]

plt.figure(figsize=(7, 4))
plt.bar(methods, fps)
plt.ylabel("FPS")
plt.title("Performance comparison of processing methods")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "performance_fps_comparison.png"), dpi=300)
plt.close()


# =========================================================
# 2) Average time per frame Plot
# =========================================================

avg_time = [
    0.0130,
    0.0071,
    0.0834,
    0.0390,
]

plt.figure(figsize=(7, 4))
plt.bar(methods, avg_time)
plt.ylabel("Average time per frame [s]")
plt.title("Average processing time per frame")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "processing_time_comparison.png"), dpi=300)
plt.close()


# =========================================================
# 3) IoU Plot
# =========================================================

regions = [
    "Inner pipe",
    "Particle bed",
]

mean_iou = [
    0.0101,
    0.0085,
]

plt.figure(figsize=(5, 4))
plt.bar(regions, mean_iou)
plt.ylabel("Mean IoU")
plt.title("IoU evaluation of OpenCV segmentation")
plt.ylim(0, 0.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "iou_comparison.png"), dpi=300)
plt.close()

print("Plots saved in:", OUT_DIR)



