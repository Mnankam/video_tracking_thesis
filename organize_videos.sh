#!/bin/bash
#SBATCH --job-name=sort_videos
#SBATCH --partition=scc-cpu        
#SBATCH --time=01:00:00            
#SBATCH --mem=4G                   
#SBATCH --cpus-per-task=2          
#SBATCH --output=logs/sort_%j.out
#SBATCH --error=logs/sort_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=serge.nankam@stud.hawk.de

SRC="/scratch-scc/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/video_side"
DST="/scratch-scc/projects/mthesis_s_kouomnankam/video_tracking_thesis/data"

mkdir -p $DST/train
mkdir -p $DST/test
mkdir -p $DST/debug

for file in $SRC/*.MP4; do
    size=$(stat -c%s "$file")

    if [ $size -gt 1000000000 ]; then
        echo "Groß → TRAIN: $file"
        cp "$file" $DST/train/
    elif [ $size -gt 200000000 ]; then
        echo "Mittel → TEST: $file"
        cp "$file" $DST/test/
    else
        echo "Klein → DEBUG: $file"
        cp "$file" $DST/debug/
    fi
done
