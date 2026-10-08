#!/usr/bin/env bash
# T9.4/T9.6: train + evaluate one recipe's OOF fold set sequentially on one GPU. Waits for the
# GPU's currently-running job to report its exit marker first (never interrupts it), then runs each
# fold's train+evaluate, fail-fast. Evaluates explicitly with --split test (the genuinely held-out
# report subjects for that fold), not run_stage5_queue.py's hardcoded --split val -- applying the
# LODO/loso_met split-role lesson directly rather than reusing the generic queue runner.
#
# Usage: run_oof_queue.sh <recipe: seg|cls> <device: cuda:0|cuda:1> <wait_log> <wait_marker_prefix>
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
RECIPE="$1"
DEVICE="$2"
WAIT_LOG="$3"
WAIT_MARKER_PREFIX="$4"
LOG="logs_oof_${RECIPE}_queue.log"
DIR=artifacts/private/oof_folds

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "OOF_${RECIPE}_QUEUE FAILED: $1"; echo "OOF_${RECIPE}_QUEUE_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== waiting for $WAIT_LOG to report ${WAIT_MARKER_PREFIX}_EXIT= ==="
while ! grep -q "${WAIT_MARKER_PREFIX}_EXIT=" "$WAIT_LOG" 2>/dev/null; do
  sleep 30
done
exit_line=$(grep "${WAIT_MARKER_PREFIX}_EXIT=" "$WAIT_LOG" | tail -1)
log "prerequisite finished: $exit_line"
if [[ "$exit_line" != *"${WAIT_MARKER_PREFIX}_EXIT=0" ]]; then
  fail "prerequisite job ($WAIT_LOG) did not exit 0 -- not starting OOF queue on a possibly-contended GPU"
fi

ids=$($PY -c "
import json
reg = json.load(open('$DIR/oof_registry.json'))
for r in reg:
    if r['recipe'] == '$RECIPE':
        print(r['id'])
")

for id in $ids; do
  config="$DIR/${id}.yaml"
  log "=== $id: train ==="
  $PY scripts/train.py --config "$config" --device "$DEVICE" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "$id train"

  log "=== $id: evaluate (--split test, the held-out report subjects for this fold) ==="
  $PY scripts/evaluate.py --config "$config" --checkpoint "outputs/$id/checkpoints/best.pt" \
    --split test --device "$DEVICE" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "$id evaluate"

  log "=== $id complete ==="
done

log "=== OOF ${RECIPE} queue complete (${ids//$'\n'/ }) ==="
echo "OOF_${RECIPE}_QUEUE_EXIT=0" | tee -a "$LOG"
