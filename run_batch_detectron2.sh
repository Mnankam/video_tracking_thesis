#!/bin/bash

#SBATCH --job-name=detectron2_full
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/detectron2_full_%j.out
#SBATCH --error=logs/detectron2_full_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=serge.nankam@stud.hawk.de

set -euo pipefail


# =============================================================================
# OpenCV / FFmpeg
# =============================================================================

export OPENCV_FFMPEG_READ_ATTEMPTS=131072
export APPTAINERENV_OPENCV_FFMPEG_READ_ATTEMPTS=131072


# =============================================================================
# Paths
# =============================================================================

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"

PROJECT="$HOME/projects/video_tracking_thesis"

DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/vide_sides"

BASE_CONFIG="$PROJECT/configs/config_detectron2.yaml"

EXPERIMENT_NAME="Detectron2_GPU_FullFrames"

OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/${EXPERIMENT_NAME}"


# =============================================================================
# Directories
# =============================================================================

mkdir -p "$OUT"
mkdir -p "$OUT/logs"
mkdir -p "$OUT/configs"

mkdir -p "$PROJECT/logs"

cd "$PROJECT"


# =============================================================================
# Sanity checks
# =============================================================================

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found:"
    echo "$CONTAINER"
    exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config not found:"
    echo "$BASE_CONFIG"
    exit 1
fi

if [ ! -d "$DATA" ]; then
    echo "ERROR: Video directory not found:"
    echo "$DATA"
    exit 1
fi


# =============================================================================
# Video list
# =============================================================================

shopt -s nullglob

VIDEOS=("$DATA"/*.MP4)

if [ "${#VIDEOS[@]}" -eq 0 ]; then
    echo "ERROR: No MP4 files found in:"
    echo "$DATA"
    exit 1
fi

echo "============================================================"
echo "DETECTRON2 FULL-FRAME BATCH"
echo "============================================================"
echo "Experiment : $EXPERIMENT_NAME"
echo "Videos     : ${#VIDEOS[@]}"
echo "Input      : $DATA"
echo "Output     : $OUT"
echo "============================================================"


# =============================================================================
# Process videos
# =============================================================================

VIDEO_COUNTER=0
SUCCESS_COUNTER=0
FAIL_COUNTER=0


for video in "${VIDEOS[@]}"; do

    VIDEO_COUNTER=$((VIDEO_COUNTER + 1))

    name=$(basename "$video" .MP4)

    echo
    echo "============================================================"
    echo "[$VIDEO_COUNTER/${#VIDEOS[@]}] Processing $name"
    echo "============================================================"


    CONFIG_OUT="$OUT/configs/config_${name}.yaml"

    DETECTRON2_OUT="$OUT/${name}_detectron2_results.csv"

    FRAME_OUT="$OUT/${name}_detectron2_results_frames.csv"

    BENCHMARK_OUT="$OUT/${name}_detectron2_results_benchmark.csv"

    DEBUG_OUT="$OUT/debug_${name}"

    PLOT_DIR="$OUT/plots_${name}"

    ANALYSIS_DIR="$OUT/analysis_${name}"


    mkdir -p \
        "$DEBUG_OUT" \
        "$PLOT_DIR" \
        "$ANALYSIS_DIR"


    # =========================================================================
    # Generate video-specific configuration
    #
    # In addition to replacing paths, explicitly force a complete run:
    #
    #     start_frame: 0
    #     end_frame: null
    #
    # =========================================================================

    sed \
        -e "s|^video_path:.*|video_path: $video|" \
        -e "s|^detectron2_output_csv:.*|detectron2_output_csv: $DETECTRON2_OUT|" \
        -e "s|^detectron2_debug_dir:.*|detectron2_debug_dir: $DEBUG_OUT|" \
        -e "s|^start_frame:.*|start_frame: 0|" \
        -e "s|^end_frame:.*|end_frame: null|" \
        "$BASE_CONFIG" \
        > "$CONFIG_OUT"


    # =========================================================================
    # Verify generated configuration
    # =========================================================================

    echo "Generated configuration:"
    grep -E \
        "^(video_path|start_frame|end_frame|detectron2_output_csv|detectron2_debug_dir):" \
        "$CONFIG_OUT" \
        || true


    # =========================================================================
    # Detectron2 inference
    # =========================================================================

    echo
    echo "Running full-frame Detectron2 inference..."


    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.detectron2_inference \
            --config "$CONFIG_OUT" \
            --output-csv "$DETECTRON2_OUT" \
            --debug-dir "$DEBUG_OUT" \
        > "$OUT/logs/${name}_detectron2.log" 2>&1
    then

        echo "Detectron2 inference finished: $name"

    else

        echo "ERROR: Detectron2 inference failed: $name"
        echo "Log:"
        echo "$OUT/logs/${name}_detectron2.log"

        FAIL_COUNTER=$((FAIL_COUNTER + 1))

        continue
    fi


    # =========================================================================
    # Validate expected files
    # =========================================================================

    if [ ! -f "$FRAME_OUT" ]; then
        echo "ERROR: Missing frame-level output:"
        echo "$FRAME_OUT"

        FAIL_COUNTER=$((FAIL_COUNTER + 1))

        continue
    fi


    if [ ! -f "$BENCHMARK_OUT" ]; then
        echo "ERROR: Missing benchmark output:"
        echo "$BENCHMARK_OUT"

        FAIL_COUNTER=$((FAIL_COUNTER + 1))

        continue
    fi


    # =========================================================================
    # Analysis
    # =========================================================================

    echo "Running Detectron2 analysis..."


    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.analyse_detectron2_results \
            --csv "$DETECTRON2_OUT" \
            --frame-csv "$FRAME_OUT" \
            --out-dir "$ANALYSIS_DIR" \
        > "$OUT/logs/${name}_detectron2_analysis.log" 2>&1
    then

        echo "Analysis finished: $name"

    else

        echo "WARNING: Analysis failed: $name"

    fi


    # =========================================================================
    # Plots
    # =========================================================================

    echo "Running Detectron2 plots..."


    if apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.plot_detectron2_results \
            --csv "$DETECTRON2_OUT" \
            --out-dir "$PLOT_DIR" \
        > "$OUT/logs/${name}_detectron2_plot.log" 2>&1
    then

        echo "Plots finished: $name"

    else

        echo "WARNING: Plot generation failed: $name"

    fi


    # =========================================================================
    # Print benchmark result
    # =========================================================================

    echo
    echo "Benchmark result for $name:"
    cat "$BENCHMARK_OUT"


    SUCCESS_COUNTER=$((SUCCESS_COUNTER + 1))

    echo
    echo "Completed: $name"

done


# =============================================================================
# Final batch summary
# =============================================================================

echo
echo "============================================================"
echo "DETECTRON2 FULL-FRAME BATCH FINISHED"
echo "============================================================"
echo "Experiment             : $EXPERIMENT_NAME"
echo "Videos discovered      : ${#VIDEOS[@]}"
echo "Videos attempted       : $VIDEO_COUNTER"
echo "Successful video runs  : $SUCCESS_COUNTER"
echo "Failed video runs      : $FAIL_COUNTER"
echo "Output directory       : $OUT"
echo "============================================================"