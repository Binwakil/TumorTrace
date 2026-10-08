#!/usr/bin/env bash
# Evaluate the matched 15% modality-dropout checkpoint under each forced-missing sequence.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace

PY="conda run -n llava python"
BASE=outputs/m5_modality_dropout_d3/resolved_config.json
CHECKPOINT=outputs/m5_modality_dropout_d3/checkpoints/best.pt
CONFIG_DIR=artifacts/private/m5_robustness/configs
CASE_DIR=artifacts/private/m5_robustness/cases
LOG=logs_m5_modality_dropout_missing_sequences.log

mkdir -p "$CONFIG_DIR" "$CASE_DIR"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() {
  log "M5_MODALITY_DROPOUT_MISSING_SEQUENCES FAILED: $1"
  echo "M5_MODALITY_DROPOUT_MISSING_SEQUENCES_EXIT=1" | tee -a "$LOG"
  exit 1
}

for modality in t1n t1c t2w t2f; do
  $PY scripts/materialize_matched_config.py \
    --base "$BASE" \
    --override "project.name=TumorTrust-modality-dropout-missing-${modality}-development" \
    --override "project.output_dir=outputs/m5_moddrop_missing_${modality}" \
    --override project.operation=evaluate \
    --override "data.force_missing_modality=${modality}" \
    --allow-changed data.force_missing_modality \
    --output "$CONFIG_DIR/moddrop_missing_${modality}.yaml" \
    2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "${modality} config materialization"
done

run_queue() {
  local gpu=$1
  shift
  local modality
  for modality in "$@"; do
    log "=== modality-dropout checkpoint, forced missing ${modality}, GPU ${gpu} ==="
    CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/evaluate.py \
      --config "$CONFIG_DIR/moddrop_missing_${modality}.yaml" \
      --checkpoint "$CHECKPOINT" --split val --device cuda \
      --case-output "$CASE_DIR/moddrop_missing_${modality}_val_cases.json" \
      2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} == 0 ]] || return 1
  done
}

run_queue 0 t1n t1c &
pid0=$!
run_queue 1 t2w t2f &
pid1=$!

wait "$pid0" || fail "GPU 0 queue"
wait "$pid1" || fail "GPU 1 queue"

log "=== all modality-dropout missing-sequence cross-evaluations complete ==="
echo "M5_MODALITY_DROPOUT_MISSING_SEQUENCES_EXIT=0" | tee -a "$LOG"
