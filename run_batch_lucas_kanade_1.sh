#!/bin/bash
#SBATCH --job-name=internal_validation
#SBATCH --partition=scc-gpu        
#SBATCH --gres=gpu:A100:1
#SBATCH --time=16:00:00            
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/internal_val_%j.out
#SBATCH --error=logs/internal_val_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=serge.nankam@stud.hawk.de

set -euo pipefail

# =========================================================
# OpenCV / FFmpeg video decoding configuration
# =========================================================

export OPENCV_FFMPEG_READ_ATTEMPTS=131072
export APPTAINERENV_OPENCV_FFMPEG_READ_ATTEMPTS=131072

# =========================================================
# Projektpfade
# =========================================================

CONTAINER="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/containers/detectron2.sif"

PROJECT="${HOME}/projects/video_tracking_thesis"

DATA="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/Validation"

BASE_CONFIG="${PROJECT}/configs/config.yaml"

EXPERIMENT_NAME="results_internal_validierung"

OUT="/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/video_tracking_thesis/data/Validation/${EXPERIMENT_NAME}"

# =========================================================
# Steuerung der Verarbeitungsschritte
# =========================================================

RUN_PIPELINE=true
RUN_LUCAS_KANADE=true
RUN_PLOTS=true
RUN_ANALYSIS=true
RUN_ANIMATION=true
RUN_VISUALIZATION=true


ONLY_VIDEO=""

# Bestehende Ergebnisdateien überschreiben?
OVERWRITE=false

# =========================================================
# Hilfsfunktionen
# =========================================================

log()
{
    local message="$1"
    printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message"
}


fail()
{
    local message="$1"
    log "FEHLER: ${message}"
    exit 1
}


require_file()
{
    local path="$1"
    local description="$2"

    if [[ ! -f "$path" ]]; then
        fail "${description} nicht gefunden: ${path}"
    fi
}


require_dir()
{
    local path="$1"
    local description="$2"

    if [[ ! -d "$path" ]]; then
        fail "${description} nicht gefunden: ${path}"
    fi
}


run_apptainer()
{
    apptainer exec \
        -B /mnt/ceph-hdd:/mnt/ceph-hdd \
        -B "${PROJECT}:${PROJECT}" \
        "${CONTAINER}" \
        "$@"
}

check_output_collision()
{
    local path="$1"

    if [[ -e "$path" && "$OVERWRITE" != true ]]; then
        fail "Ausgabe existiert bereits: ${path}

Setze OVERWRITE=true oder verwende einen neuen EXPERIMENT_NAME."
    fi
}


create_configs()
{
    local video="$1"
    local pipeline_config="$2"
    local lk_config="$3"
    local results_out="$4"
    local summary_out="$5"
    local debug_out="$6"
    local lk_out="$7"

    # -----------------------------------------------------
    # Config für Segmentierungs-/Tracking-Pipeline
    #
    # Wichtig:
    # enable_optical_flow wird deaktiviert, damit die Pipeline
    # Lucas-Kanade nicht zusätzlich ausführt.
    # -----------------------------------------------------

    sed \
        -e "s|^video_path:.*|video_path: ${video}|" \
        -e "s|^output_csv:.*|output_csv: ${results_out}|" \
        -e "s|^summary_csv:.*|summary_csv: ${summary_out}|" \
        -e "s|^debug_dir:.*|debug_dir: ${debug_out}|" \
        -e "s|^optical_flow_csv:.*|optical_flow_csv: ${lk_out}|" \
        -e "s|^enable_optical_flow:.*|enable_optical_flow: false|" \
        "${BASE_CONFIG}" \
        > "${pipeline_config}"

    # -----------------------------------------------------
    # Config für eigenständige Lucas-Kanade-Ausführung
    # -----------------------------------------------------

    sed \
        -e "s|^video_path:.*|video_path: ${video}|" \
        -e "s|^output_csv:.*|output_csv: ${results_out}|" \
        -e "s|^summary_csv:.*|summary_csv: ${summary_out}|" \
        -e "s|^debug_dir:.*|debug_dir: ${debug_out}|" \
        -e "s|^optical_flow_csv:.*|optical_flow_csv: ${lk_out}|" \
        -e "s|^enable_optical_flow:.*|enable_optical_flow: true|" \
        "${BASE_CONFIG}" \
        > "${lk_config}"
}


validate_generated_config()
{
    local config_path="$1"
    local expected_video="$2"
    local expected_lk_status="$3"

    python - "${config_path}" "${expected_video}" "${expected_lk_status}" <<'PY'
import pathlib
import sys

import yaml


config_path = pathlib.Path(sys.argv[1])
expected_video = pathlib.Path(sys.argv[2])
expected_lk_status = sys.argv[3].lower() == "true"

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

if not isinstance(config, dict):
    raise SystemExit(f"Ungültige YAML-Konfiguration: {config_path}")

actual_video = pathlib.Path(str(config.get("video_path", "")))

if actual_video != expected_video:
    raise SystemExit(
        "Falscher video_path in Config:\n"
        f"  erwartet: {expected_video}\n"
        f"  erhalten: {actual_video}"
    )

actual_lk_status = bool(config.get("enable_optical_flow", False))

if actual_lk_status != expected_lk_status:
    raise SystemExit(
        "Falscher enable_optical_flow-Wert:\n"
        f"  erwartet: {expected_lk_status}\n"
        f"  erhalten: {actual_lk_status}"
    )

points = config.get("optical_flow_points", [])

if not isinstance(points, list) or len(points) == 0:
    raise SystemExit(
        f"Keine optical_flow_points in Config gefunden: {config_path}"
    )

print(
    f"Config geprüft: {config_path.name} | "
    f"video={actual_video.name} | "
    f"enable_optical_flow={actual_lk_status} | "
    f"num_points={len(points)}"
)
PY
}


# =========================================================
# Vorprüfungen
# =========================================================

require_file "${CONTAINER}" "Apptainer-Container"
require_dir "${PROJECT}" "Projektverzeichnis"
require_dir "${DATA}" "Video-Datenverzeichnis"
require_file "${BASE_CONFIG}" "Basis-Config"

command -v apptainer >/dev/null 2>&1 \
    || fail "apptainer ist nicht verfügbar."

mkdir -p \
    "${OUT}" \
    "${OUT}/logs" \
    "${OUT}/configs"

cd "${PROJECT}"

log "Projekt: ${PROJECT}"
log "Daten: ${DATA}"
log "Experiment: ${EXPERIMENT_NAME}"
log "Ausgabe: ${OUT}"

# =========================================================
# Videoliste erzeugen
# =========================================================

declare -a VIDEOS=()

if [[ -n "${ONLY_VIDEO}" ]]; then

    target_video="${DATA}/${ONLY_VIDEO}.MP4"

    require_file "${target_video}" "Gewähltes Video"

    VIDEOS+=("${target_video}")

else

    shopt -s nullglob

    for video in "${DATA}"/*.MP4 "${DATA}"/*.mp4; do
        VIDEOS+=("${video}")
    done

    shopt -u nullglob

    if [[ ${#VIDEOS[@]} -eq 0 ]]; then
        fail "Keine MP4-Dateien in ${DATA} gefunden."
    fi

fi

log "Anzahl Videos: ${#VIDEOS[@]}"

# =========================================================
# Verarbeitung
# =========================================================

for video in "${VIDEOS[@]}"; do

    filename="$(basename "${video}")"
    name="${filename%.*}"

    PIPELINE_CONFIG="${OUT}/configs/config_${name}_pipeline.yaml"
    LK_CONFIG="${OUT}/configs/config_${name}_lk.yaml"

    RESULTS_OUT="${OUT}/${name}_results.csv"
    SUMMARY_OUT="${OUT}/${name}_summary.csv"

    LK_OUT="${OUT}/${name}_lucas_kanade.csv"
    LK_BENCHMARK="${OUT}/${name}_lucas_kanade_benchmark.csv"
    LK_INITIAL_POINTS="${OUT}/${name}_lucas_kanade_initial_points.png"

    LK_PLOT_DIR="${OUT}/plots_${name}"
    LK_ANALYSIS_DIR="${OUT}/analysis_${name}"

    LK_ANIMATION="${OUT}/${name}_lucas_kanade_animation.avi"

    DEBUG_OUT="${OUT}/debug_${name}"

    PIPELINE_LOG="${OUT}/logs/${name}_pipeline.log"
    LK_LOG="${OUT}/logs/${name}_lucas_kanade.log"
    PLOT_LOG="${OUT}/logs/${name}_lucas_kanade_plot.log"
    ANALYSIS_LOG="${OUT}/logs/${name}_lucas_kanade_analysis.log"
    ANIMATION_LOG="${OUT}/logs/${name}_lucas_kanade_animation.log"
    VISUALIZATION_LOG="${OUT}/logs/${name}_visualization.log"

    mkdir -p \
        "${DEBUG_OUT}" \
        "${LK_PLOT_DIR}" \
        "${LK_ANALYSIS_DIR}"

    log "========================================================="
    log "Verarbeite Video: ${name}"
    log "Video-Datei: ${video}"
    log "========================================================="

    # -----------------------------------------------------
    # Kollisionsprüfung
    # -----------------------------------------------------

    if [[ "${RUN_LUCAS_KANADE}" == true ]]; then
        check_output_collision "${LK_OUT}"
        check_output_collision "${LK_BENCHMARK}"
        check_output_collision "${LK_INITIAL_POINTS}"
    fi

    if [[ "${RUN_ANIMATION}" == true ]]; then
        check_output_collision "${LK_ANIMATION}"
    fi

    # -----------------------------------------------------
    # Repository-spezifische Configs erzeugen
    # -----------------------------------------------------

    create_configs \
        "${video}" \
        "${PIPELINE_CONFIG}" \
        "${LK_CONFIG}" \
        "${RESULTS_OUT}" \
        "${SUMMARY_OUT}" \
        "${DEBUG_OUT}" \
        "${LK_OUT}"

    validate_generated_config \
        "${PIPELINE_CONFIG}" \
        "${video}" \
        false

    validate_generated_config \
        "${LK_CONFIG}" \
        "${video}" \
        true

    # =====================================================
    # 1. Segmentierung und reguläres Tracking
    # =====================================================

    if [[ "${RUN_PIPELINE}" == true ]]; then

        log "Pipeline wird ausgeführt."

        if run_apptainer \
            python -m src.pipeline \
            --config "${PIPELINE_CONFIG}" \
            > "${PIPELINE_LOG}" 2>&1; then

            log "Pipeline abgeschlossen: ${name}"

        else

            log "Pipeline fehlgeschlagen: ${name}"
            log "Logdatei: ${PIPELINE_LOG}"
            exit 1

        fi

    else

        log "Pipeline übersprungen."

    fi

    # =====================================================
    # 2. Lucas-Kanade Optical Flow
    # =====================================================

    if [[ "${RUN_LUCAS_KANADE}" == true ]]; then

        log "Lucas-Kanade Optical Flow wird ausgeführt."

        if run_apptainer \
            python -m src.optical_flow_test \
            --config "${LK_CONFIG}" \
            --output-csv "${LK_OUT}" \
            > "${LK_LOG}" 2>&1; then

            log "Lucas-Kanade abgeschlossen: ${name}"

        else

            log "Lucas-Kanade fehlgeschlagen: ${name}"
            log "Logdatei: ${LK_LOG}"
            exit 1

        fi

        require_file "${LK_OUT}" "Lucas-Kanade CSV"
        require_file "${LK_BENCHMARK}" "Lucas-Kanade Benchmark"
        require_file "${LK_INITIAL_POINTS}" "Lucas-Kanade Initialpunkte-Bild"

    else

        log "Lucas-Kanade übersprungen."

        require_file "${LK_OUT}" \
            "Vorhandene Lucas-Kanade CSV für Folgeschritte"

    fi

    # =====================================================
    # 3. Lucas-Kanade Plots
    # =====================================================

    if [[ "${RUN_PLOTS}" == true ]]; then

        log "Lucas-Kanade Plots werden erzeugt."

        if run_apptainer \
            python -m src.plot_optical_flow \
            --csv "${LK_OUT}" \
            --out-dir "${LK_PLOT_DIR}" \
            > "${PLOT_LOG}" 2>&1; then

            log "Lucas-Kanade Plots abgeschlossen: ${name}"

        else

            log "Lucas-Kanade Plot-Erzeugung fehlgeschlagen: ${name}"
            log "Logdatei: ${PLOT_LOG}"
            exit 1

        fi

    else

        log "Lucas-Kanade Plots übersprungen."

    fi

    # =====================================================
    # 4. Lucas-Kanade Analyse
    # =====================================================

    if [[ "${RUN_ANALYSIS}" == true ]]; then

        log "Lucas-Kanade Analyse wird ausgeführt."

        if run_apptainer \
            python -m src.analyse_optical_flow \
            --csv "${LK_OUT}" \
            --out-dir "${LK_ANALYSIS_DIR}" \
            > "${ANALYSIS_LOG}" 2>&1; then

            log "Lucas-Kanade Analyse abgeschlossen: ${name}"

        else

            log "Lucas-Kanade Analyse fehlgeschlagen: ${name}"
            log "Logdatei: ${ANALYSIS_LOG}"
            exit 1

        fi

    else

        log "Lucas-Kanade Analyse übersprungen."

    fi

    # =====================================================
    # 5. Lucas-Kanade Animation
    # =====================================================

    if [[ "${RUN_ANIMATION}" == true ]]; then

        log "Lucas-Kanade Animation wird erzeugt."

        if run_apptainer \
            python -m src.animate_optical_flow_cv \
            --video "${video}" \
            --csv "${LK_OUT}" \
            --out "${LK_ANIMATION}" \
            --config "${LK_CONFIG}" \
            --start-frame 1 \
            --end-frame 110000 \
            > "${ANIMATION_LOG}" 2>&1; then

            log "Lucas-Kanade Animation abgeschlossen: ${name}"

        else

            log "Lucas-Kanade Animation fehlgeschlagen: ${name}"
            log "Logdatei: ${ANIMATION_LOG}"
            exit 1

        fi

    else

        log "Lucas-Kanade Animation übersprungen."

    fi

    # =====================================================
    # 6. Visualisierung der OpenCV-Segmentierung
    # =====================================================

    if [[ "${RUN_VISUALIZATION}" == true ]]; then

        log "Segmentierungsvisualisierung wird ausgeführt."

        if run_apptainer \
            python -m src.visualization \
            --config "${PIPELINE_CONFIG}" \
            > "${VISUALIZATION_LOG}" 2>&1; then

            log "Visualisierung abgeschlossen: ${name}"

        else

            # Visualisierung ist kein zwingender Bestandteil der
            # Lucas-Kanade-Datenerzeugung. Deshalb wird hier nicht
            # der gesamte Batch abgebrochen.
            log "WARNUNG: Visualisierung fehlgeschlagen: ${name}"
            log "Logdatei: ${VISUALIZATION_LOG}"

        fi

    else

        log "Visualisierung übersprungen."

    fi

    # =====================================================
    # Ergebnisübersicht
    # =====================================================

    log "Video abgeschlossen: ${name}"
    log "Lucas-Kanade CSV: ${LK_OUT}"
    log "Benchmark: ${LK_BENCHMARK}"
    log "Initialpunkte: ${LK_INITIAL_POINTS}"
    log "Plots: ${LK_PLOT_DIR}"
    log "Analyse: ${LK_ANALYSIS_DIR}"
    log "Animation: ${LK_ANIMATION}"

done

log "Alle ausgewählten Lucas-Kanade-Videos wurden verarbeitet."