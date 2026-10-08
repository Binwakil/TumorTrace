#!/usr/bin/env bash
# Overnight Priority 2 (GPU 1): the authoritative matched separate-encoder control.
# Waits for the already-running exploratory v2 MET-holdout job in tumortrust-after-m2 to finish on
# its own (never interrupted or overwritten) before starting, per the standing GPU-1-single-job rule.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:1
LOG=logs_overnight_gpu1.log
DIR=artifacts/private/overnight

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

fail() {
  log "GPU1_QUEUE FAILED: $1"
  echo "GPU1_QUEUE_EXIT=1" | tee -a "$LOG"
  exit 1
}

log "=== GPU1 queue: waiting for the exploratory v2 MET-holdout job (logs_after_m2.log) to finish ==="
while ! grep -q "AFTER_M2_CHAIN_EXIT=\|TRAIN FAILED\|EVALUATE FAILED" logs_after_m2.log 2>/dev/null; do
  sleep 30
done
log "exploratory job reached a terminal state: $(tail -3 logs_after_m2.log | tr '\n' ' ')"
log "(exploratory result recorded as exploratory regardless of outcome; proceeding to Priority 2 either way -- it does not gate the separate-encoder control)"

log "--- materialize matched separate-encoder config (from outputs/M0/resolved_config.json) ---"
$PY scripts/materialize_matched_config.py \
  --base outputs/M0/resolved_config.json \
  --override project.name=TumorTrust-VLM-separate-encoder-matched-control \
  --override project.output_dir=outputs/separate_encoder_matched_control \
  --override model.architecture=separate_encoder_segresnet \
  --allow-changed model.architecture \
  --output "$DIR/separate_encoder_matched_control.yaml" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "matched separate-encoder config-diff assertion"

log "--- train + evaluate the matched separate-encoder control (idempotent, restart-safe queue) ---"
cat > "$DIR/gpu1_registry.json" << 'EOF'
{
  "experiments": [
    {
      "id": "separate_encoder_matched_control",
      "config": "artifacts/private/overnight/separate_encoder_matched_control.yaml"
    }
  ]
}
EOF
$PY scripts/run_stage5_queue.py --registry "$DIR/gpu1_registry.json" \
  --status "$DIR/gpu1_status.json" --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "matched separate-encoder control train/evaluate queue"

log "=== GPU1 queue complete ==="
echo "GPU1_QUEUE_EXIT=0" | tee -a "$LOG"
