#!/usr/bin/env bash
# T6.7 matched modality-dropout training, launched only after inference-only robustness queues.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_m5_modality_dropout_training.log
CONFIG=artifacts/private/m5_robustness/configs/modality_dropout.yaml
OUTPUT=outputs/m5_modality_dropout_d3

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "M5_MODALITY_DROPOUT FAILED: $1"; echo "M5_MODALITY_DROPOUT_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== waiting for both inference-only M5 robustness queues ==="
while true; do
  missing=$(grep 'M5_MISSING_SEQUENCES_EXIT=' logs_m5_missing_sequences.log 2>/dev/null | tail -1)
  shifts=$(grep 'M5_CONTROLLED_SHIFTS_EXIT=' logs_m5_controlled_shifts.log 2>/dev/null | tail -1)
  [[ -n "$missing" && -n "$shifts" ]] && break
  sleep 30
done
[[ "$missing" == *"=0" ]] || fail "missing-sequence queue did not complete"
[[ "$shifts" == *"=0" ]] || fail "controlled-shift queue did not complete"

log "=== materialize exact matched 15% modality-dropout training config ==="
$PY scripts/materialize_matched_config.py \
  --base outputs/D3_zscore/resolved_config.json \
  --override project.name=TumorTrust-modality-dropout-development \
  --override project.output_dir="$OUTPUT" \
  --override project.operation=train \
  --override data.missing_modality_probability=0.15 \
  --allow-changed data.missing_modality_probability \
  --output "$CONFIG" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "config materialization"

log "=== train exact matched modality-dropout condition ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/train.py --config "$CONFIG" --device cuda \
  2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "training"

log "=== evaluate modality-dropout condition on frozen development validation ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate.py --config "$OUTPUT/resolved_config.json" \
  --checkpoint "$OUTPUT/checkpoints/best.pt" --split val --device cuda \
  --case-output artifacts/private/m5_robustness/cases/modality_dropout_val_cases.json \
  2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "evaluation"
log "=== matched modality-dropout training/evaluation complete ==="
echo "M5_MODALITY_DROPOUT_EXIT=0" | tee -a "$LOG"
