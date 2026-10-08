#!/usr/bin/env bash
# Overnight Priority 1 (GPU 0): the authoritative matched MET source-holdout classifier.
# Fail-fast: any nonzero exit stops this branch immediately and leaves completed results in place.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:0
LOG=logs_overnight_gpu0.log
DIR=artifacts/private/overnight
V2_MANIFEST=artifacts/private/cross_evaluations/loso_met_met__trainingdata2_v2_development_split.json

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

fail() {
  log "GPU0_QUEUE FAILED: $1"
  echo "GPU0_QUEUE_EXIT=1" | tee -a "$LOG"
  exit 1
}

log "=== GPU0 queue start ==="

log "--- validate v2 MET-holdout manifest ---"
$PY scripts/validate_split_manifest.py --manifest "$V2_MANIFEST" \
  --report "$DIR/met_v2_manifest_validation.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "manifest validation"

log "--- materialize matched MET-holdout classifier config (from outputs/C0/resolved_config.json) ---"
$PY scripts/materialize_matched_config.py \
  --base outputs/C0/resolved_config.json \
  --override project.name=TumorTrust-VLM-met-holdout-matched-classifier \
  --override project.output_dir=outputs/met_holdout_matched_classifier \
  --override data.split_manifest="$V2_MANIFEST" \
  --allow-changed data.split_manifest \
  --output "$DIR/met_holdout_matched_classifier.yaml" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "matched-classifier config-diff assertion"

log "--- materialize C0-on-v2-panel reference config (from outputs/C0/resolved_config.json) ---"
$PY scripts/materialize_matched_config.py \
  --base outputs/C0/resolved_config.json \
  --override project.name=TumorTrust-VLM-c0-reference-on-met-v2-panel \
  --override project.output_dir=outputs/c0_reference_on_met_v2_panel \
  --override data.split_manifest="$V2_MANIFEST" \
  --allow-changed data.split_manifest \
  --output "$DIR/c0_reference_on_met_v2_panel.yaml" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "reference-config config-diff assertion"

log "--- evaluate existing C0 checkpoint on v2 test panel (source-seen reference; separate output dir, C0 untouched) ---"
[[ -f outputs/C0/checkpoints/best.pt ]] || fail "outputs/C0/checkpoints/best.pt missing"
$PY scripts/evaluate.py --config "$DIR/c0_reference_on_met_v2_panel.yaml" \
  --checkpoint outputs/C0/checkpoints/best.pt --split test --device "$DEVICE" \
  --case-output artifacts/private/c0_reference_on_met_v2_panel_test_cases.json 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "C0 reference evaluation"

log "--- train + evaluate the matched MET-holdout classifier (idempotent, restart-safe queue) ---"
cat > "$DIR/gpu0_registry.json" << 'EOF'
{
  "experiments": [
    {
      "id": "met_holdout_matched_classifier",
      "config": "artifacts/private/overnight/met_holdout_matched_classifier.yaml"
    }
  ]
}
EOF
$PY scripts/run_stage5_queue.py --registry "$DIR/gpu0_registry.json" \
  --status "$DIR/gpu0_status.json" --device "$DEVICE" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "matched MET-holdout classifier train/evaluate queue"

log "=== GPU0 queue complete ==="
echo "GPU0_QUEUE_EXIT=0" | tee -a "$LOG"
