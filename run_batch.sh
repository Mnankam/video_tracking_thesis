#!/bin/bash
set -euo pipefail

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"
PROJECT="$HOME/projects/video_tracking_thesis"
DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/test"
OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/batch_inner_pipe_cv0_27"

mkdir -p "$OUT/logs" "$OUT/configs"

cd "$PROJECT"

for video in "$DATA"/*.MP4; do
    [ -e "$video" ] || { echo "Keine MP4-Dateien in $DATA gefunden."; exit 1; }

    name=$(basename "$video" .MP4)

    CONFIG_OUT="$OUT/configs/config_${name}.yaml"
    RESULTS_OUT="$OUT/${name}_results.csv"
    SUMMARY_OUT="$OUT/${name}_summary.csv"
    OPTICAL_FLOW_OUT="$OUT/${name}_optical_flow.csv"
    OPTICAL_FLOW_ANIMATION="$OUT/${name}_optical_flow_animation.mp4"
    DEBUG_OUT="$OUT/debug_${name}"
    OPTICAL_FLOW_PLOT_DIR="$OUT/plots_${name}"

    echo "======================================"
    echo "Processing: $name"
    echo "Video: $video"
    echo "======================================"

    mkdir -p "$DEBUG_OUT" "$OPTICAL_FLOW_PLOT_DIR"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^output_csv:.*|output_csv: $RESULTS_OUT|; \
         s|^summary_csv:.*|summary_csv: $SUMMARY_OUT|; \
         s|^debug_dir:.*|debug_dir: $DEBUG_OUT|; \
         s|^optical_flow_csv:.*|optical_flow_csv: $OPTICAL_FLOW_OUT|" \
         configs/config.yaml > "$CONFIG_OUT"

    if [ -f "$RESULTS_OUT" ]; then
        echo "Result already exists for $name, skipping pipeline."
    else
        echo "Running pipeline..."
        if apptainer exec \
            -B /mnt/ceph-hdd:/mnt/ceph-hdd \
            -B "$PROJECT":"$PROJECT" \
            "$CONTAINER" \
            python -m src.pipeline --config "$CONFIG_OUT" \
            > "$OUT/logs/${name}_pipeline.log" 2>&1; then
            echo "Pipeline finished: $name"
        else
            echo "Pipeline failed for $name. Check log:"
            echo "$OUT/logs/${name}_pipeline.log"
            continue
        fi
    fi

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
            echo "Optical flow failed for $name. Check log:"
            echo "$OUT/logs/${name}_optical_flow.log"
        fi
    fi

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
        echo "Optical flow plots failed for $name. Check log:"
        echo "$OUT/logs/${name}_optical_flow_plot.log"
    fi

    echo "Running optical flow animation..."
    if apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.animate_optical_flow \
            --video "$video" \
            --csv "$OPTICAL_FLOW_OUT" \
            --out "$OPTICAL_FLOW_ANIMATION" \
            --start-frame 0 \
            --end-frame 10000 \
        > "$OUT/logs/${name}_optical_flow_animation.log" 2>&1; then
        echo "Optical flow animation finished: $name"
    else
        echo "Optical flow animation failed for $name. Check log:"
        echo "$OUT/logs/${name}_optical_flow_animation.log"
    fi

    echo "Running visualization..."
    if apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.visualization --config "$CONFIG_OUT" \
        > "$OUT/logs/${name}_visualization.log" 2>&1; then
        echo "Visualization finished: $name"
    else
        echo "Visualization failed for $name. Check log:"
        echo "$OUT/logs/${name}_visualization.log"
    fi

    echo "Done: $name"
done

echo "======================================"
echo "All videos processed."
echo "======================================"