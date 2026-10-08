#!/usr/bin/env bash
# T10.3: compact text-only language-prior baseline on frozen OOF reporter records.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_reporter_r1.log
OUTPUT=outputs/reporter_text_only_compact

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "REPORTER_R1 FAILED: $1"; echo "REPORTER_R1_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== waiting for frozen OOF reporter records ==="
while ! grep -q 'OOF_REPORTER_RECORDS_EXIT=' logs_oof_reporter_records.log 2>/dev/null; do sleep 30; done
marker=$(grep 'OOF_REPORTER_RECORDS_EXIT=' logs_oof_reporter_records.log | tail -1)
[[ "$marker" == *"=0" ]] || fail "OOF reporter records did not complete"

log "=== text-only compact reporter train ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_reporter.py \
  --train-records artifacts/private/reporter_records_train.json \
  --val-records artifacts/private/reporter_records_val.json \
  --vocabulary artifacts/private/reporter_word_vocabulary_v1.json \
  --output "$OUTPUT" --mode text_only --epochs 30 --batch-size 8 \
  --seed 20260812 --maximum-length 256 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "training"

log "=== text-only compact reporter evaluate on report-validation ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_reporter.py \
  --records artifacts/private/reporter_records_val.json \
  --checkpoint "$OUTPUT/best.pt" --output "$OUTPUT/val_predictions.json" \
  --mode text_only --batch-size 8 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "evaluation"
log "=== text-only compact reporter complete ==="
echo "REPORTER_R1_EXIT=0" | tee -a "$LOG"
