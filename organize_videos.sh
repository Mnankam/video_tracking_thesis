#!/bin/bash

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
