#!/usr/bin/env bash
# T10.7 R5: predicted visual tokens plus oracle development evidence (diagnostic only).
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_reporter_oracle_upper_bound.log
OUTPUT=outputs/reporter_oracle_evidence_compact

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "REPORTER_ORACLE FAILED: $1"; echo "REPORTER_ORACLE_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== waiting for compact predicted-evidence matrix ==="
while ! grep -q 'REPORTER_COMPACT_MATRIX_EXIT=' logs_reporter_compact_matrix.log 2>/dev/null; do sleep 15; done
marker=$(grep 'REPORTER_COMPACT_MATRIX_EXIT=' logs_reporter_compact_matrix.log | tail -1)
[[ "$marker" == *"=0" ]] || fail "compact matrix did not complete"

log "=== build development-only oracle-evidence records ==="
$PY scripts/build_oracle_reporter_records.py \
  --train-records artifacts/private/reporter_records_train.json \
  --val-records artifacts/private/reporter_records_val.json \
  --output-train artifacts/private/reporter_records_train_oracle.json \
  --output-val artifacts/private/reporter_records_val_oracle.json 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "oracle record generation"

log "=== train visual plus oracle-evidence diagnostic upper bound ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_reporter.py \
  --train-records artifacts/private/reporter_records_train_oracle.json \
  --val-records artifacts/private/reporter_records_val_oracle.json \
  --vocabulary artifacts/private/reporter_word_vocabulary_v1.json \
  --output "$OUTPUT" --mode full --epochs 30 --batch-size 8 \
  --seed 20260812 --maximum-length 256 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "training"

for decoding in unconstrained constrained; do
  extra=()
  [[ "$decoding" == constrained ]] && extra+=(--schema-constrained)
  log "=== oracle upper bound: ${decoding} evaluation ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_reporter.py \
    --records artifacts/private/reporter_records_val_oracle.json \
    --checkpoint "$OUTPUT/best.pt" \
    --output "$OUTPUT/val_predictions_${decoding}.json" \
    --mode full --batch-size 8 "${extra[@]}" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "${decoding} evaluation"
done
log "=== oracle-evidence reporter upper bound complete ==="
echo "REPORTER_ORACLE_EXIT=0" | tee -a "$LOG"
