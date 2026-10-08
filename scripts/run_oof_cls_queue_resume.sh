#!/usr/bin/env bash
# T9.4: resume the cls-recipe OOF queue at fold_4 + reportval, skipping the failed fold_3 (being
# retried separately on the other GPU via run_oof_cls_fold3_ampoff.sh). Reuses the same
# train-then-evaluate(--split test) pattern as run_oof_queue.sh but only for the two remaining ids,
# since that script's registry loop has no skip mechanism and folds 0-2 are already complete.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:1
LOG=logs_oof_cls_queue.log
DIR=artifacts/private/oof_folds

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "OOF_cls_QUEUE FAILED: $1"; echo "OOF_cls_QUEUE_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== resuming cls OOF queue at fold_4 (skipping failed fold_3, retried separately on GPU 0) ==="

for id in oof_cls_fold_4_development oof_cls_reportval_development; do
  config="$DIR/${id}.yaml"
  log "=== $id: train ==="
  $PY scripts/train.py --config "$config" --device "$DEVICE" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "$id train"

  log "=== $id: evaluate (--split test) ==="
  $PY scripts/evaluate.py --config "$config" --checkpoint "outputs/$id/checkpoints/best.pt" \
    --split test --device "$DEVICE" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "$id evaluate"

  log "=== $id complete ==="
done

log "=== OOF cls queue resume complete (fold_4, reportval) -- fold_3 tracked separately ==="
echo "OOF_cls_QUEUE_RESUME_EXIT=0" | tee -a "$LOG"
