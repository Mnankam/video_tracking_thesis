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

EXPERIMENT_NAME="Farneback_Dense_CPU"

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

    FARNEBACK_OUT="$OUT/${name}_farneback_dense.csv"

    DEBUG_OUT="$OUT/debug_${name}"
    FARNEBACK_ANALYSIS_DIR="$OUT/analysis_${name}"
    VISUALIZATION_OUT="$OUT/${name}_visualization.mp4"

    echo "======================================"
    echo "Processing: $name"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Video: $video"
    echo "======================================"

    mkdir -p \
        "$DEBUG_OUT" \
        "$FARNEBACK_ANALYSIS_DIR"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^output_csv:.*|output_csv: $RESULTS_OUT|; \
         s|^summary_csv:.*|summary_csv: $SUMMARY_OUT|; \
         s|^debug_dir:.*|debug_dir: $DEBUG_OUT|; \
         s|^optical_flow_csv:.*|optical_flow_csv: $FARNEBACK_OUT|" \
         configs/config.yaml > "$CONFIG_OUT"

    # =========================================================
    # Pipeline: Segmentierung + Tracking
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
    # Farneback Dense Optical Flow
    # =========================================================

    if [ -f "$FARNEBACK_OUT" ]; then

        echo "Farneback already exists for $name, skipping."

    else

        echo "Running Farneback dense optical flow..."

        if apptainer exec \
            -B /mnt/ceph-hdd:/mnt/ceph-hdd \
            -B "$PROJECT":"$PROJECT" \
            "$CONTAINER" \
            python -m src.optical_flow_farneback \
                --config "$CONFIG_OUT" \
                --output-csv "$FARNEBACK_OUT" \
            > "$OUT/logs/${name}_farneback.log" 2>&1; then

            echo "Farneback finished: $name"

        else

            echo "Farneback failed for $name"
            echo "$OUT/logs/${name}_farneback.log"

        fi
    fi

    # =========================================================
    # Analyse Farneback
    # =========================================================
    # Hinweis:
    # analyse_optical_flow.py ist ursprünglich für x/y/point_id gedacht.
    # Für Farneback nutzt du später besser ein eigenes Analyse-Skript.
    # Deshalb wird hier erstmal nur die CSV erzeugt.

    echo "Farneback CSV ready:"
    echo "$FARNEBACK_OUT"

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

echo "======================================"
echo "All videos processed."
echo "Experiment: $EXPERIMENT_NAME"
echo "======================================"