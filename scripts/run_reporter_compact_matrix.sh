#!/usr/bin/env bash
# T10 compact R2-R4 matched reporter ablations on frozen OOF development records.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_reporter_compact_matrix.log

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "REPORTER_COMPACT_MATRIX FAILED: $1"; echo "REPORTER_COMPACT_MATRIX_EXIT=1" | tee -a "$LOG"; exit 1; }

if ! grep -q 'OOF_REPORTER_RECORDS_EXIT=0' logs_oof_reporter_records.log 2>/dev/null; then
  fail "frozen OOF reporter records are not complete"
fi

run_one() {
  local run_id="$1"
  local mode="$2"
  local output="outputs/reporter_${run_id}_compact"
  log "=== ${run_id} ${mode}: train ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_reporter.py \
    --train-records artifacts/private/reporter_records_train.json \
    --val-records artifacts/private/reporter_records_val.json \
    --vocabulary artifacts/private/reporter_word_vocabulary_v1.json \
    --output "$output" --mode "$mode" --epochs 30 --batch-size 8 \
    --seed 20260812 --maximum-length 256 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "${run_id} training"

  for decoding in unconstrained constrained; do
    extra=()
    [[ "$decoding" == constrained ]] && extra+=(--schema-constrained)
    log "=== ${run_id} ${mode}: ${decoding} evaluation ==="
    CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_reporter.py \
      --records artifacts/private/reporter_records_val.json \
      --checkpoint "$output/best.pt" \
      --output "$output/val_predictions_${decoding}.json" \
      --mode "$mode" --batch-size 8 "${extra[@]}" 2>&1 | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} == 0 ]] || fail "${run_id} ${decoding} evaluation"
  done
}

run_one evidence_only evidence_only
run_one visual_only visual_only
run_one visual_evidence_full full
log "=== compact R2-R4 reporter matrix complete ==="
echo "REPORTER_COMPACT_MATRIX_EXIT=0" | tee -a "$LOG"
