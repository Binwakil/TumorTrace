#!/usr/bin/env bash
# Evaluate the frozen D3 checkpoint under the four preregistered controlled shifts.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_m5_controlled_shifts.log
DIR=artifacts/private/m5_robustness
CHECKPOINT=outputs/D3_zscore/checkpoints/best.pt

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "M5_CONTROLLED_SHIFTS FAILED: $1"; echo "M5_CONTROLLED_SHIFTS_EXIT=1" | tee -a "$LOG"; exit 1; }

for shift in noise intensity bias_field resolution; do
  log "=== controlled ${shift} shift: frozen 248-case development validation ==="
  CUDA_VISIBLE_DEVICES=1 $PY scripts/evaluate.py \
    --config "$DIR/configs/shift_${shift}.yaml" \
    --checkpoint "$CHECKPOINT" --split val --device cuda \
    --case-output "$DIR/cases/shift_${shift}_val_cases.json" \
    2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "${shift} evaluation"
done
log "=== all controlled-shift evaluations complete ==="
echo "M5_CONTROLLED_SHIFTS_EXIT=0" | tee -a "$LOG"
