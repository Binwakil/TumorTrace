#!/usr/bin/env bash
# T6.5: LOSO segmentation on the corrected v2 MET-holdout manifest (identical to the frozen
# segmentation-only D3 z-score baseline except data.split_manifest, whitelist-asserted). This
# manifest uses LODO-style role semantics (test = the genuine held-out panel; val = an
# in-distribution early-stopping subsample) -- evaluate explicitly with --split test, not the
# generic queue runner's hardcoded --split val (the exact mistake found and fixed for the LODO
# folds and the original loso_met fold earlier this session).
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:1
LOG=logs_loso_met_segmentation.log
CONFIG=artifacts/private/overnight/loso_met_segmentation_matched.yaml
ID=loso_met_segmentation_matched

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "LOSO_MET_SEG FAILED: $1"; echo "LOSO_MET_SEG_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== $ID: train ==="
$PY scripts/train.py --config "$CONFIG" --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "train"

log "=== $ID: evaluate (--split test -- the genuine held-out-source-branch + GLI/MEN panel) ==="
$PY scripts/evaluate.py --config "$CONFIG" --checkpoint "outputs/$ID/checkpoints/best.pt" \
  --split test --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "evaluate"

log "=== $ID complete ==="
echo "LOSO_MET_SEG_EXIT=0" | tee -a "$LOG"
