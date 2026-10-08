#!/usr/bin/env bash
# Priority 2 retry (user-authorized 2026-08-15): the matched separate-encoder control, identical to
# the frozen fixed-weight multitask recipe except model.architecture and training.amp (whitelist-
# asserted). Diagnostics (scripts/probe_separate_encoder_nan.py + a 15-epoch amp-off smoke check)
# supported amp:false as the likely fix; this is the full authoritative 300-epoch confirmation.
# On success, proceeds directly to Priority 4 (development-only LODO). On a second failure, stops and
# reports -- no further automatic escalation.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:1
LOG=logs_overnight_gpu1_retry.log
DIR=artifacts/private/overnight

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() {
  log "GPU1_RETRY FAILED: $1"
  echo "GPU1_RETRY_EXIT=1" | tee -a "$LOG"
  exit 1
}

log "=== GPU1 retry start: matched separate-encoder control, amp:false ==="

cat > "$DIR/gpu1_retry_registry.json" << 'EOF'
{
  "experiments": [
    {
      "id": "separate_encoder_matched_control_ampoff",
      "config": "artifacts/private/overnight/separate_encoder_matched_control_ampoff.yaml"
    }
  ]
}
EOF
$PY scripts/run_stage5_queue.py --registry "$DIR/gpu1_retry_registry.json" \
  --status "$DIR/gpu1_retry_status.json" --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "matched separate-encoder control (amp:false) train/evaluate queue"

log "=== GPU1 retry complete: separate-encoder control succeeded under amp:false ==="
echo "GPU1_RETRY_EXIT=0" | tee -a "$LOG"

log "=== Priority 4: regenerate + validate LODO-GLI/MEN/MET from outputs/D3_zscore/resolved_config.json ==="
$PY scripts/materialize_lodo_matched.py 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "LODO matched materialization/validation"

log "--- queue LODO-GLI + LODO-MEN sequentially on GPU 0 (smaller training pools) ---"
cat > "$DIR/gpu0_lodo_registry.json" << 'EOF'
{
  "experiments": [
    {"id": "lodo_gli_development", "config": "artifacts/private/cross_evaluations/lodo_gli_development.yaml"},
    {"id": "lodo_men_development", "config": "artifacts/private/cross_evaluations/lodo_men_development.yaml"}
  ]
}
EOF
$PY scripts/run_stage5_queue.py --registry "$DIR/gpu0_lodo_registry.json" \
  --status "$DIR/gpu0_lodo_status.json" --device cuda:0 2>&1 | tee -a "$LOG" &
GPU0_LODO_PID=$!

log "--- queue LODO-MET on GPU 1 (largest training pool) ---"
cat > "$DIR/gpu1_lodo_registry.json" << 'EOF'
{
  "experiments": [
    {"id": "lodo_met_development", "config": "artifacts/private/cross_evaluations/lodo_met_development.yaml"}
  ]
}
EOF
$PY scripts/run_stage5_queue.py --registry "$DIR/gpu1_lodo_registry.json" \
  --status "$DIR/gpu1_lodo_status.json" --device "$DEVICE" 2>&1 | tee -a "$LOG"
GPU1_LODO_RC=$?

wait "$GPU0_LODO_PID"
GPU0_LODO_RC=$?

if [[ "$GPU0_LODO_RC" != 0 || "$GPU1_LODO_RC" != 0 ]]; then
  fail "LODO queue(s) failed: gpu0_rc=$GPU0_LODO_RC gpu1_rc=$GPU1_LODO_RC"
fi

log "=== Priority 4 (development-only LODO-GLI/MEN/MET) complete ==="
echo "PRIORITY4_LODO_EXIT=0" | tee -a "$LOG"
