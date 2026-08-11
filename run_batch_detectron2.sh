#!/bin/bash
set -euo pipefail

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"
PROJECT="$HOME/projects/video_tracking_thesis"
DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/Validation"

EXPERIMENT_NAME="Detectron2_GPU_Baseline2"
OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/${EXPERIMENT_NAME}"

mkdir -p "$OUT/logs" "$OUT/configs"

cd "$PROJECT"

for video in "$DATA"/*.MP4; do
    [ -e "$video" ] || { echo "Keine MP4-Dateien gefunden."; exit 1; }

    name=$(basename "$video" .MP4)

    CONFIG_OUT="$OUT/configs/config_${name}.yaml"
    DETECTRON2_OUT="$OUT/${name}_detectron2_results.csv"
    DETECTRON2_ANIMATION="$OUT/${name}_detectron2_animation.avi"

    DEBUG_OUT="$OUT/debug_${name}"
    PLOT_DIR="$OUT/plots_${name}"
    ANALYSIS_DIR="$OUT/analysis_${name}"

    mkdir -p "$DEBUG_OUT" "$PLOT_DIR" "$ANALYSIS_DIR"

    echo "======================================"
    echo "Processing: $name"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Video: $video"
    echo "======================================"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^detectron2_output_csv:.*|detectron2_output_csv: $DETECTRON2_OUT|; \
         s|^detectron2_debug_dir:.*|detectron2_debug_dir: $DEBUG_OUT|" \
         configs/config_detectron2.yaml > "$CONFIG_OUT"

    echo "Running Detectron2 inference..."

    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.detectron2_inference \
            --config "$CONFIG_OUT" \
            --output-csv "$DETECTRON2_OUT" \
            --debug-dir "$DEBUG_OUT" \
        > "$OUT/logs/${name}_detectron2.log" 2>&1; then

        echo "Detectron2 finished: $name"

    else
        echo "Detectron2 failed for $name"
        echo "$OUT/logs/${name}_detectron2.log"
        continue
    fi

    echo "Running Detectron2 plots..."

    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.plot_detectron2_results \
            --csv "$DETECTRON2_OUT" \
            --out-dir "$PLOT_DIR" \
        > "$OUT/logs/${name}_detectron2_plot.log" 2>&1; then

        echo "Detectron2 plots finished: $name"

    else
        echo "Detectron2 plots failed for $name"
        echo "$OUT/logs/${name}_detectron2_plot.log"
    fi

    echo "Running Detectron2 analysis..."

    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.analyse_detectron2_results \
            --csv "$DETECTRON2_OUT" \
            --out-dir "$ANALYSIS_DIR" \
        > "$OUT/logs/${name}_detectron2_analysis.log" 2>&1; then

        echo "Detectron2 analysis finished: $name"

    else
        echo "Detectron2 analysis failed for $name"
        echo "$OUT/logs/${name}_detectron2_analysis.log"
    fi

    echo "Running Detectron2 animation..."

    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.animate_optical_flow_detectron2 \
            --config "$CONFIG_OUT" \
            --csv "$DETECTRON2_OUT" \
            --out "$DETECTRON2_ANIMATION" \
            --start-frame 1 \
            --end-frame 300 \
            --draw-reference-rois \
        > "$OUT/logs/${name}_detectron2_animation.log" 2>&1; then

        echo "Detectron2 animation finished: $name"

    else
        echo "Detectron2 animation failed for $name"
        echo "$OUT/logs/${name}_detectron2_animation.log"
    fi

    echo "Done: $name"

done

echo "======================================"
echo "All Detectron2 videos processed."
echo "Experiment: $EXPERIMENT_NAME"
echo "======================================"
