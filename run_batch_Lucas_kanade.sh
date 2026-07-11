#!/bin/bash
set -euo pipefail
# =========================================================
# OpenCV / FFmpeg video decoding configuration
# =========================================================

export OPENCV_FFMPEG_READ_ATTEMPTS=131072
export APPTAINERENV_OPENCV_FFMPEG_READ_ATTEMPTS=131072

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"
PROJECT="$HOME/projects/video_tracking_thesis"
DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/train"

EXPERIMENT_NAME="Lucas_Kanade_CPU7"
OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/${EXPERIMENT_NAME}"

mkdir -p "$OUT/logs" "$OUT/configs"
cd "$PROJECT"

for video in "$DATA"/*.MP4; do
    [ -e "$video" ] || { echo "Keine MP4-Dateien gefunden."; exit 1; }

    name=$(basename "$video" .MP4)

    CONFIG_OUT="$OUT/configs/config_${name}.yaml"
    RESULTS_OUT="$OUT/${name}_results.csv"
    SUMMARY_OUT="$OUT/${name}_summary.csv"
    LK_OUT="$OUT/${name}_lucas_kanade.csv"
    LK_PLOT_DIR="$OUT/plots_${name}"
    LK_ANALYSIS_DIR="$OUT/analysis_${name}"
    LK_ANIMATION="$OUT/${name}_lucas_kanade_animation.avi"
    DEBUG_OUT="$OUT/debug_${name}"

    mkdir -p "$DEBUG_OUT" "$LK_PLOT_DIR" "$LK_ANALYSIS_DIR"

    echo "======================================"
    echo "Processing: $name"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "======================================"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^output_csv:.*|output_csv: $RESULTS_OUT|; \
         s|^summary_csv:.*|summary_csv: $SUMMARY_OUT|; \
         s|^debug_dir:.*|debug_dir: $DEBUG_OUT|; \
         s|^optical_flow_csv:.*|optical_flow_csv: $LK_OUT|" \
         configs/config.yaml > "$CONFIG_OUT"

    # =========================================================
    # Pipeline: Segmentierung + Tracking
    # =========================================================

    echo "Running pipeline..."
    apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.pipeline --config "$CONFIG_OUT" \
        > "$OUT/logs/${name}_pipeline.log" 2>&1

    # =========================================================
    # Lukas Kanade Optical Flow
    # =========================================================

    echo "Running Lucas-Kanade optical flow..."
    apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.optical_flow_test \
            --config "$CONFIG_OUT" \
            --output-csv "$LK_OUT" \
        > "$OUT/logs/${name}_lucas_kanade.log" 2>&1

    # =========================================================
    # Lukas Kanade  Plot
    # =========================================================

    echo "Running Lucas-Kanade plots..."
    apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.plot_optical_flow \
            --csv "$LK_OUT" \
            --out-dir "$LK_PLOT_DIR" \
        > "$OUT/logs/${name}_lucas_kanade_plot.log" 2>&1

    # =========================================================
    # Lukas Kanade  analysis
    # =========================================================

    echo "Running Lucas-Kanade analysis..."
    apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.analyse_optical_flow \
            --csv "$LK_OUT" \
            --out-dir "$LK_ANALYSIS_DIR" \
        > "$OUT/logs/${name}_lucas_kanade_analysis.log" 2>&1

    # =========================================================
    # Lukas Kanade  Animation
    # =========================================================

    echo "Running Lucas-Kanade animation..."
    apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.animate_optical_flow_cv \
            --video "$video" \
            --csv "$LK_OUT" \
            --out "$LK_ANIMATION" \
            --config "$CONFIG_OUT" \
            --start-frame 1 \
            --end-frame 110000 \
        > "$OUT/logs/${name}_lucas_kanade_animation.log" 2>&1

    # =========================================================
    # Visualization der OpenCV-Segmentierung
    # =========================================================

    echo "Running visualization..."

    if apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.visualization \
            --config "$CONFIG_OUT" \
        > "$OUT/logs/${name}_visualization.log" 2>&1; then

        echo "Visualization finished: $name"

    else

        echo "Visualization failed for $name"
        echo "$OUT/logs/${name}_visualization.log"

    fi

    echo "Done: $name"

done

echo "All Lucas-Kanade videos processed."