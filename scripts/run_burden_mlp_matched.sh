#!/usr/bin/env bash
# T7.5: matched auxiliary burden-MLP ablation (identical to the frozen segmentation-only D3 z-score
# baseline except model.predict_burden/model.lambda_burden, whitelist-asserted). Standard master
# split manifest, so --split val is the correct evaluation role (no LODO-style role mismatch here).
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:0
LOG=logs_burden_mlp_matched.log
CONFIG=artifacts/private/overnight/burden_mlp_matched.yaml
ID=burden_mlp_matched

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "BURDEN_MLP FAILED: $1"; echo "BURDEN_MLP_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== $ID: train ==="
$PY scripts/train.py --config "$CONFIG" --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "train"

log "=== $ID: evaluate (--split val, standard master-split panel) ==="
$PY scripts/evaluate.py --config "$CONFIG" --checkpoint "outputs/$ID/checkpoints/best.pt" \
  --split val --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "evaluate"

log "=== $ID complete ==="
echo "BURDEN_MLP_EXIT=0" | tee -a "$LOG"
