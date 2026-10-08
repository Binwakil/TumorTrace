#!/usr/bin/env bash
# Export development-only OOF segmentation/visual evidence for T10 reporter training.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_oof_feature_seg.log
DIR=artifacts/private/oof_features

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "OOF_FEATURE_SEG FAILED: $1"; echo "OOF_FEATURE_SEG_EXIT=1" | tee -a "$LOG"; exit 1; }

mkdir -p "$DIR"
log "=== segmentation-recipe OOF feature export start ==="
for suffix in fold_0 fold_1 fold_2 fold_3 fold_4 reportval; do
  id="oof_seg_${suffix}_development"
  config="artifacts/private/oof_folds/${id}.yaml"
  checkpoint="outputs/${id}/checkpoints/best.pt"
  validation="artifacts/private/oof_folds/${id}_split_validation.json"
  output="$DIR/${id}.json"
  [[ -f "$checkpoint" ]] || fail "$id missing checkpoint"
  [[ "$(jq -r '.passed' "$validation")" == "true" ]] || fail "$id manifest validation"
  log "=== $id extract --split test ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/extract_core_features.py \
    --config "$config" --checkpoint "$checkpoint" --split test --output "$output" \
    2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "$id extraction"
done
log "=== segmentation-recipe OOF feature export complete ==="
echo "OOF_FEATURE_SEG_EXIT=0" | tee -a "$LOG"
