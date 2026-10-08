#!/usr/bin/env bash
# Evaluate the frozen D3 checkpoint under one forced-missing MRI sequence at a time.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_m5_missing_sequences.log
DIR=artifacts/private/m5_robustness
CHECKPOINT=outputs/D3_zscore/checkpoints/best.pt

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "M5_MISSING_SEQUENCES FAILED: $1"; echo "M5_MISSING_SEQUENCES_EXIT=1" | tee -a "$LOG"; exit 1; }

for modality in t1n t1c t2w t2f; do
  log "=== forced missing ${modality}: frozen 248-case development validation ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py \
    --config "$DIR/configs/missing_${modality}.yaml" \
    --checkpoint "$CHECKPOINT" --split val --device cuda \
    --case-output "$DIR/cases/missing_${modality}_val_cases.json" \
    2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "${modality} evaluation"
done
log "=== all forced-missing-sequence evaluations complete ==="
echo "M5_MISSING_SEQUENCES_EXIT=0" | tee -a "$LOG"
