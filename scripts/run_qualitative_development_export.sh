#!/usr/bin/env bash
# Export and plot prespecified development-only qualitative cases on GPU 1 after OOF extraction.
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
LOG=logs_qualitative_development_export.log

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "QUALITATIVE_EXPORT FAILED: $1"; echo "QUALITATIVE_EXPORT_EXIT=1" | tee -a "$LOG"; exit 1; }

log "=== waiting for GPU-1 OOF classification feature queue ==="
while ! grep -q 'OOF_FEATURE_CLS_EXIT=' logs_oof_feature_cls.log 2>/dev/null; do sleep 30; done
marker=$(grep 'OOF_FEATURE_CLS_EXIT=' logs_oof_feature_cls.log | tail -1)
[[ "$marker" == *"=0" ]] || fail "classification feature queue did not complete"

log "=== export nine frozen development qualitative cases ==="
CUDA_VISIBLE_DEVICES=1 $PY scripts/export_qualitative_development_cases.py --device cuda \
  2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "case export"

log "=== render manuscript Figure 3 ==="
$PY scripts/plot_qualitative_segmentation_evidence.py 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "figure render"
log "=== qualitative development export complete ==="
echo "QUALITATIVE_EXPORT_EXIT=0" | tee -a "$LOG"
