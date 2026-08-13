#!/bin/bash
#SBATCH --job-name=cuda_flow_gpu1
#SBATCH --partition=scc-gpu        
#SBATCH --gres=gpu:A100:1
#SBATCH --time=08:00:00            
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cuda_flow_%j.out
#SBATCH --error=logs/cuda_flow_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=serge.nankam@stud.hawk.de
set -euo pipefail

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"
PROJECT="$HOME/projects/video_tracking_thesis"
DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/test"

EXPERIMENT_NAME="CUDA_Optical_Flow_GPU1"
OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/outputs/${EXPERIMENT_NAME}"

mkdir -p "$OUT/logs" "$OUT/configs"
cd "$PROJECT"

for video in "$DATA"/*.MP4; do
    [ -e "$video" ] || { echo "Keine MP4-Dateien gefunden."; exit 1; }

    name=$(basename "$video" .MP4)

    CONFIG_OUT="$OUT/configs/config_${name}.yaml"
    RESULTS_OUT="$OUT/${name}_results.csv"
    SUMMARY_OUT="$OUT/${name}_summary.csv"
    CUDA_OUT="$OUT/${name}_cuda_optical_flow.csv"
    CUDA_ANALYSIS_DIR="$OUT/analysis_${name}"
    CUDA_PLOT_DIR="$OUT/plots_${name}"
    CUDA_ANIMATION="$OUT/${name}_cuda_animation.avi"
    DEBUG_OUT="$OUT/debug_${name}"

    mkdir -p "$DEBUG_OUT" "$CUDA_ANALYSIS_DIR" "$CUDA_PLOT_DIR"

    echo "======================================"
    echo "Processing: $name"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "Video: $video"
    echo "======================================"

    sed "s|^video_path:.*|video_path: $video|; \
         s|^output_csv:.*|output_csv: $RESULTS_OUT|; \
         s|^summary_csv:.*|summary_csv: $SUMMARY_OUT|; \
         s|^debug_dir:.*|debug_dir: $DEBUG_OUT|; \
         s|^optical_flow_csv:.*|optical_flow_csv: $CUDA_OUT|" \
         configs/config.yaml > "$CONFIG_OUT"

    echo "Running pipeline..."
    apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.pipeline \
            --config "$CONFIG_OUT" \
        > "$OUT/logs/${name}_pipeline.log" 2>&1

    echo "Running CUDA optical flow..."
    apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.optical_flow_cuda \
            --config "$CONFIG_OUT" \
            --output-csv "$CUDA_OUT" \
        > "$OUT/logs/${name}_cuda.log" 2>&1

    echo "Running CUDA plots..."
    apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.plot_optical_flow_cuda \
            --csv "$CUDA_OUT" \
            --out-dir "$CUDA_PLOT_DIR" \
        > "$OUT/logs/${name}_cuda_plot.log" 2>&1

    echo "Running CUDA analysis..."
    apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.analyse_optical_flow_cuda \
            --csv "$CUDA_OUT" \
            --out-dir "$CUDA_ANALYSIS_DIR" \
        > "$OUT/logs/${name}_cuda_analysis.log" 2>&1

    echo "Running CUDA animation..."
    apptainer exec --nv \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "$PROJECT":"$PROJECT" \
        "$CONTAINER" \
        python -m src.animate_optical_flow_cv_cuda \
            --config "$CONFIG_OUT" \
            --out "$CUDA_ANIMATION" \
            --start-frame 1 \
            --end-frame 300 \
        > "$OUT/logs/${name}_cuda_animation.log" 2>&1

    echo "Done: $name"
done

echo "======================================"
echo "All CUDA Optical Flow videos processed."
echo "Experiment: $EXPERIMENT_NAME"
echo "======================================"