#!/bin/bash
#SBATCH --job-name=video_metadata
#SBATCH --partition=scc-cpu        
#SBATCH --time=02:00:00            
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/metadata_%j.out
#SBATCH --error=logs/metadata_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=serge.nankam@stud.hawk.de

set -u

VIDEO_DIR="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/video_sides/video_side"
OUTPUT_CSV="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/video_sides/video_metadata.csv"

# ---------------------------------------------------------------------------
# Check requirements
# ---------------------------------------------------------------------------

if [[ ! -d "$VIDEO_DIR" ]]; then
    echo "ERROR: Video directory does not exist:"
    echo "       $VIDEO_DIR"
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ERROR: ffprobe was not found."
    exit 1
fi

# ---------------------------------------------------------------------------
# Create CSV only if it does not already exist
# ---------------------------------------------------------------------------

if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "video_id,filename,creation_time,duration_s,fps,width,height,frames,frame_count_source,file_size_bytes" \
        > "$OUTPUT_CSV"

    echo "Created new metadata file:"
    echo "$OUTPUT_CSV"
else
    echo "Existing metadata file found."
    echo "Already processed videos will be skipped."
fi

echo
echo "Scanning videos in:"
echo "$VIDEO_DIR"
echo

processed_count=0
skipped_count=0
failed_count=0

# ---------------------------------------------------------------------------
# Process all video files
# ---------------------------------------------------------------------------

while IFS= read -r -d '' video; do

    filename=$(basename "$video")
    video_id="${filename%.*}"

    # -----------------------------------------------------------------------
    # Resume check
    #
    # Check whether this exact video_id already occurs in the first CSV column.
    # -----------------------------------------------------------------------

    if awk -F',' -v id="\"$video_id\"" '
        NR > 1 && $1 == id {
            found=1
            exit
        }
        END {
            exit !found
        }
    ' "$OUTPUT_CSV"; then

        echo "SKIP:      $filename (already processed)"
        ((skipped_count+=1))
        continue
    fi

    echo "PROCESSING: $filename"

    # -----------------------------------------------------------------------
    # Container / file metadata
    # -----------------------------------------------------------------------

    creation_time=$(ffprobe \
        -v error \
        -show_entries format_tags=creation_time \
        -of default=noprint_wrappers=1:nokey=1 \
        "$video" 2>/dev/null | head -n 1)

    # Fallback: creation_time may be stored in the video stream
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

    # -----------------------------------------------------------------------
    # Video stream metadata
    # -----------------------------------------------------------------------

    stream_data=$(ffprobe \
        -v error \
        -select_streams v:0 \
        -show_entries stream=width,height,avg_frame_rate,nb_frames \
        -of csv=p=0 \
        "$video" 2>/dev/null)

    IFS=',' read -r width height fps_fraction frames <<< "$stream_data"

    # -----------------------------------------------------------------------
    # FPS conversion
    # Example: 200/1 -> 200.000000
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Determine frame count
    # -----------------------------------------------------------------------

    if [[ -n "${frames:-}" && "$frames" != "N/A" ]]; then

        frame_count_source="ffprobe_nb_frames"

    else

        if [[ -n "${duration:-}" && -n "${fps:-}" ]]; then

            frames=$(awk \
                -v d="$duration" \
                -v f="$fps" \
                'BEGIN {printf "%.0f", d*f}')

            frame_count_source="estimated_duration_x_fps"

        else

            frames=""
            frame_count_source="NA"

        fi
    fi

    # -----------------------------------------------------------------------
    # Check whether essential metadata could be extracted
    # -----------------------------------------------------------------------

    if [[ -z "${width:-}" || -z "${height:-}" ]]; then

        echo "FAILED:     $filename (video stream metadata unavailable)"
        ((failed_count+=1))
        continue

    fi

    # -----------------------------------------------------------------------
    # Replace missing optional values by NA
    # -----------------------------------------------------------------------

    creation_time="${creation_time:-NA}"
    duration="${duration:-NA}"
    fps="${fps:-NA}"
    width="${width:-NA}"
    height="${height:-NA}"
    frames="${frames:-NA}"
    file_size="${file_size:-NA}"

    # -----------------------------------------------------------------------
    # Append completed video to CSV
    #
    # IMPORTANT:
    # The row is written only AFTER processing has completed.
    # Therefore an interrupted video is not incorrectly marked as finished.
    # -----------------------------------------------------------------------

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

    echo "DONE:       $filename"

    ((processed_count+=1))

done < <(
    find "$VIDEO_DIR" \
        -maxdepth 1 \
        -type f \
        \( -iname "*.mp4" -o -iname "*.mov" \) \
        -print0 |
    sort -z
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo

echo "============================================================"
echo "Metadata extraction completed"
echo "============================================================"
echo "Newly processed : $processed_count"
echo "Already existing: $skipped_count"
echo "Failed          : $failed_count"
echo
echo "Metadata file:"
echo "$OUTPUT_CSV"
echo "Metadata written to: $OUTPUT_CSV"

