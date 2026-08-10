#!/usr/bin/env bash

set -u

VIDEO_DIR="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/video_sides/video_side"
OUTPUT_CSV="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/video_sides/video_metadata.csv"

echo "video_id,filename,creation_time,duration_s,fps,width,height,frames,frame_count_source,file_size_bytes" > "$OUTPUT_CSV"

find "$VIDEO_DIR" -maxdepth 1 -type f \( \
    -iname "*.mp4" -o \
    -iname "*.mov" \
\) -print0 | sort -z | while IFS= read -r -d '' video; do

    filename=$(basename "$video")
    video_id="${filename%.*}"

    # ------------------------------------------------------------
    # Container / file metadata
    # ------------------------------------------------------------

    creation_time=$(ffprobe \
        -v error \
        -show_entries format_tags=creation_time \
        -of default=noprint_wrappers=1:nokey=1 \
        "$video" 2>/dev/null | head -n 1)

    # Fallback: creation_time may be stored on the video stream
    if [[ -z "${creation_time:-}" ]]; then
        creation_time=$(ffprobe \
            -v error \
            -select_streams v:0 \
            -show_entries stream_tags=creation_time \
            -of default=noprint_wrappers=1:nokey=1 \
            "$video" 2>/dev/null | head -n 1)
    fi

    duration=$(ffprobe \
        -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$video" 2>/dev/null | head -n 1)

    file_size=$(ffprobe \
        -v error \
        -show_entries format=size \
        -of default=noprint_wrappers=1:nokey=1 \
        "$video" 2>/dev/null | head -n 1)

    # ------------------------------------------------------------
    # First video stream
    # ------------------------------------------------------------

    stream_data=$(ffprobe \
        -v error \
        -select_streams v:0 \
        -show_entries stream=width,height,avg_frame_rate,nb_frames \
        -of csv=p=0 \
        "$video" 2>/dev/null)

    IFS=',' read -r width height fps_fraction frames <<< "$stream_data"

    # ------------------------------------------------------------
    # Convert FPS fraction, e.g. 200/1 -> 200.000000
    # ------------------------------------------------------------

    if [[ -n "${fps_fraction:-}" && "$fps_fraction" != "0/0" ]]; then
        fps=$(awk -F/ '
            NF == 2 && $2 != 0 {
                printf "%.6f", $1 / $2
            }
            NF == 1 {
                printf "%.6f", $1
            }
        ' <<< "$fps_fraction")
    else
        fps=""
    fi

    # ------------------------------------------------------------
    # Determine frame count
    # ------------------------------------------------------------

    if [[ -n "${frames:-}" && "$frames" != "N/A" ]]; then
        frame_count_source="ffprobe_nb_frames"
    else
        if [[ -n "${duration:-}" && -n "${fps:-}" ]]; then
            frames=$(awk -v d="$duration" -v f="$fps" \
                'BEGIN {printf "%.0f", d*f}')
            frame_count_source="estimated_duration_x_fps"
        else
            frames=""
            frame_count_source="NA"
        fi
    fi

    # ------------------------------------------------------------
    # Replace missing values by NA
    # ------------------------------------------------------------

    creation_time="${creation_time:-NA}"
    duration="${duration:-NA}"
    fps="${fps:-NA}"
    width="${width:-NA}"
    height="${height:-NA}"
    frames="${frames:-NA}"
    file_size="${file_size:-NA}"

    # ------------------------------------------------------------
    # Write CSV row
    # ------------------------------------------------------------

    printf '"%s","%s","%s",%s,%s,%s,%s,%s,"%s",%s\n' \
        "$video_id" \
        "$filename" \
        "$creation_time" \
        "$duration" \
        "$fps" \
        "$width" \
        "$height" \
        "$frames" \
        "$frame_count_source" \
        "$file_size" \
        >> "$OUTPUT_CSV"

    echo "Processed: $filename"

done

echo
echo "Metadata written to: $OUTPUT_CSV"