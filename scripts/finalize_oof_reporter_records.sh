#!/usr/bin/env bash
# Compose the two OOF recipes and build frozen development reporter records.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_oof_reporter_records.log
DIR=artifacts/private/oof_features

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "OOF_REPORTER_RECORDS FAILED: $1"; echo "OOF_REPORTER_RECORDS_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== waiting for both OOF feature recipes ==="
while true; do
  seg=$(grep 'OOF_FEATURE_SEG_EXIT=' logs_oof_feature_seg.log 2>/dev/null | tail -1)
  cls=$(grep 'OOF_FEATURE_CLS_EXIT=' logs_oof_feature_cls.log 2>/dev/null | tail -1)
  if [[ -n "$seg" && -n "$cls" ]]; then break; fi
  sleep 30
done
[[ "$seg" == *"=0" ]] || fail "segmentation recipe did not complete"
[[ "$cls" == *"=0" ]] || fail "classification recipe did not complete"

seg_train=()
cls_train=()
for fold in 0 1 2 3 4; do
  seg_train+=("$DIR/oof_seg_fold_${fold}_development.json")
  cls_train+=("$DIR/oof_cls_fold_${fold}_development.json")
done

log "=== compose report-train OOF features ==="
$PY scripts/compose_reporter_features.py \
  --segmentation "${seg_train[@]}" --classification "${cls_train[@]}" \
  --output "$DIR/composed_report_train.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "report-train composition"

log "=== compose report-validation features ==="
$PY scripts/compose_reporter_features.py \
  --segmentation "$DIR/oof_seg_reportval_development.json" \
  --classification "$DIR/oof_cls_reportval_development.json" \
  --output "$DIR/composed_report_val.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "report-validation composition"

log "=== join frozen report targets ==="
$PY scripts/build_report_records.py --config outputs/D3_zscore/resolved_config.json \
  --features "$DIR/composed_report_train.json" --split train \
  --output artifacts/private/reporter_records_train.json 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "report-train record join"
$PY scripts/build_report_records.py --config outputs/D3_zscore/resolved_config.json \
  --features "$DIR/composed_report_val.json" --split val \
  --output artifacts/private/reporter_records_val.json 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "report-validation record join"

log "=== validate reporter record sets and frozen vocabulary ==="
$PY -c "import json; from tumortrust_vlm.reporting.dataset import WordTokenizer, validate_reporter_record_sets; train=json.load(open('artifacts/private/reporter_records_train.json')); val=json.load(open('artifacts/private/reporter_records_val.json')); print(validate_reporter_record_sets(train,val)); tokenizer=WordTokenizer.load('artifacts/private/reporter_word_vocabulary_v1.json'); assert len(tokenizer)==499" \
  2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "reporter record validation"
log "=== OOF reporter records complete ==="
echo "OOF_REPORTER_RECORDS_EXIT=0" | tee -a "$LOG"
