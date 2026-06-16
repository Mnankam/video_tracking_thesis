#!/bin/bash
set -euo pipefail

# =========================================================
# Container und Projektpfade
# =========================================================

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"
PROJECT="$HOME/projects/video_tracking_thesis"
DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/test"

# =========================================================
# Experiment Name 
# =========================================================

EXPERIMENT_NAME="07_Optical_Flow_Verbesserung_für_stabile_inner_pipe_tracking"

OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/${EXPERIMENT_NAME}"

mkdir -p "$OUT/logs" "$OUT/configs"

cd "$PROJECT"

for video in "$DATA"/*.MP4; do

    [ -e "$video" ] || {
        echo "Keine MP4-Dateien in $DATA gefunden."
        exit 1
    }

    name=$(basename "$video" .MP4)

    CONFIG_OUT="$OUT/configs/config_${name}.yaml"
    RESULTS_OUT="$OUT/${name}_results.csv"
    SUMMARY_OUT="$OUT/${name}_summary.csv"
    OPTICAL_FLOW_OUT="$OUT/${name}_optical_flow.csv"
    OPTICAL_FLOW_ANIMATION="$OUT/${name}_optical_flow_animation.avi"

    DEBUG_OUT="$OUT/debug_${name}"
    OPTICAL_FLOW_PLOT_DIR="$OUT/plots_${name}"
    OPTICAL_FLOW_ANALYSIS_DIR="$OUT/analysis_${name}"

    echo "======================================"
    echo "Processing: $name"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Video: $video"
    echo "======================================"

    mkdir -p \
        "$DEBUG_OUT" \
        "$OPTICAL_FLOW_PLOT_DIR" \
        "$OPTICAL_FLOW_ANALYSIS_DIR"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^output_csv:.*|output_csv: $RESULTS_OUT|; \
         s|^summary_csv:.*|summary_csv: $SUMMARY_OUT|; \
         s|^debug_dir:.*|debug_dir: $DEBUG_OUT|; \
         s|^optical_flow_csv:.*|optical_flow_csv: $OPTICAL_FLOW_OUT|" \
         configs/config.yaml > "$CONFIG_OUT"

    # =========================================================
    # Pipeline
    # =========================================================

    if [ -f "$RESULTS_OUT" ]; then

        echo "Result already exists for $name, skipping pipeline."

    else

        echo "Running pipeline..."

        if apptainer exec \
            -B /mnt/ceph-hdd:/mnt/ceph-hdd \
            -B "$PROJECT":"$PROJECT" \
            "$CONTAINER" \
            python -m src.pipeline \
                --config "$CONFIG_OUT" \
            > "$OUT/logs/${name}_pipeline.log" 2>&1; then

            echo "Pipeline finished: $name"

        else

            echo "Pipeline failed for $name"
            echo "$OUT/logs/${name}_pipeline.log"
            continue

        fi
    fi

    # =========================================================
    # Optical Flow
    # =========================================================

    if [ -f "$OPTICAL_FLOW_OUT" ]; then

        echo "Optical flow already exists for $name, skipping."

    else

        echo "Running optical flow..."

        if apptainer exec \
            -B /mnt/ceph-hdd:/mnt/ceph-hdd \
            -B "$PROJECT":"$PROJECT" \
            "$CONTAINER" \
            python -m src.optical_flow_test \
                --config "$CONFIG_OUT" \
                --output-csv "$OPTICAL_FLOW_OUT" \
            > "$OUT/logs/${name}_optical_flow.log" 2>&1; then

            echo "Optical flow finished: $name"

        else

            echo "Optical flow failed for $name"
            echo "$OUT/logs/${name}_optical_flow.log"

        fi
    fi

    # =========================================================
    # Optical Flow Plots
    # =========================================================

    echo "Running optical flow plots..."

    if apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.plot_optical_flow \
            --csv "$OPTICAL_FLOW_OUT" \
            --out-dir "$OPTICAL_FLOW_PLOT_DIR" \
        > "$OUT/logs/${name}_optical_flow_plot.log" 2>&1; then

        echo "Optical flow plots finished: $name"

    else

        echo "Optical flow plots failed for $name"

    fi

    # =========================================================
    # Optical Flow Analysis
    # =========================================================

    echo "Running optical flow analysis..."

    if apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.analyse_optical_flow \
            --csv "$OPTICAL_FLOW_OUT" \
            --out-dir "$OPTICAL_FLOW_ANALYSIS_DIR" \
        > "$OUT/logs/${name}_optical_flow_analysis.log" 2>&1; then

        echo "Optical flow analysis finished: $name"

    else

        echo "Optical flow analysis failed for $name"

    fi

    # =========================================================
    # Optical Flow Animation
    # =========================================================

    echo "Running optical flow animation..."

    if apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.animate_optical_flow_cv \
            --video "$video" \
            --csv "$OPTICAL_FLOW_OUT" \
            --out "$OPTICAL_FLOW_ANIMATION" \
            --start-frame 1 \
            --end-frame 300 \
        > "$OUT/logs/${name}_optical_flow_animation.log" 2>&1; then

        echo "Optical flow animation finished: $name"

    else

        echo "Optical flow animation failed for $name"

    fi

    # =========================================================
    # Visualization
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

    fi

    echo "Done: $name"

done

echo "======================================"
echo "All videos processed."
echo "======================================"