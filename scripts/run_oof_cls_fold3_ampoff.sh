#!/usr/bin/env bash
# T9.4 recovery: retry oof_cls_fold_3_development with training.amp:false (matched config built
# from that run's own resolved_config.json, whitelist: training.amp only). The original failed with
# classification_loss flat-at-chance for 30 epochs then NaN; see
# artifacts/private/invalid_runs/oof_cls_fold_3_nan_20260816/INVALID_REASON.md.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:0
LOG=logs_oof_cls_fold3_ampoff.log
CONFIG=artifacts/private/oof_folds/oof_cls_fold_3_ampoff_development.yaml
ID=oof_cls_fold_3_ampoff_development

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "OOF_CLS_FOLD3_AMPOFF FAILED: $1"; echo "OOF_CLS_FOLD3_AMPOFF_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== $ID: train ==="
$PY scripts/train.py --config "$CONFIG" --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "train"

log "=== $ID: evaluate (--split test) ==="
$PY scripts/evaluate.py --config "$CONFIG" --checkpoint "outputs/$ID/checkpoints/best.pt" \
  --split test --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "evaluate"

log "=== $ID complete ==="
echo "OOF_CLS_FOLD3_AMPOFF_EXIT=0" | tee -a "$LOG"
