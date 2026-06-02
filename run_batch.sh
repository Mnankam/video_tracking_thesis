#!/bin/bash
set -euo pipefail

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"
PROJECT="$HOME/projects/video_tracking_thesis"
DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/test"
OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/batch_inner_pipe_cv0_4"

mkdir -p "$OUT/logs" "$OUT/configs"

cd "$PROJECT"

for video in "$DATA"/*.MP4; do
    [ -e "$video" ] || { echo "Keine MP4-Dateien in $DATA gefunden."; exit 1; }

    name=$(basename "$video" .MP4)

    CONFIG_OUT="$OUT/configs/config_${name}.yaml"
    RESULTS_OUT="$OUT/${name}_results.csv"
    SUMMARY_OUT="$OUT/${name}_summary.csv"
    DEBUG_OUT="$OUT/debug_${name}"

    echo "======================================"
    echo "Processing: $name"
    echo "Video: $video"
    echo "======================================"

    mkdir -p "$DEBUG_OUT"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^output_csv:.*|output_csv: $RESULTS_OUT|; \
         s|^summary_csv:.*|summary_csv: $SUMMARY_OUT|; \
         s|^debug_dir:.*|debug_dir: $DEBUG_OUT|" \
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

echo "All videos processed."
