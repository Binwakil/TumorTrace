#!/usr/bin/env bash
# M5 trust pipeline (T7.6/T7.9/T7.12): extract MC-sampled features from the frozen gradient-balanced
# multitask checkpoint (M1) on the full 248-case development validation, then run the trust/referral
# evaluator (failure-detection AUROC, risk-coverage, random-referral control).
set -uo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
DEVICE=cuda:0
LOG=logs_m5_trust_pipeline.log
DIR=artifacts/private/trust

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
fail() { log "M5_TRUST FAILED: $1"; echo "M5_TRUST_EXIT=1" | tee -a "$LOG"; exit 1; }

mkdir -p "$DIR"
log "=== extract MC-sampled (10x) features: M1 checkpoint, full val split (with calibration) ==="
# extract_core_features.py has no --device flag (hardcodes torch.device("cuda")); pin the physical
# GPU via CUDA_VISIBLE_DEVICES instead. --calibration embeds the T7.8-fitted interval bounds into
# each subject's evidence card, which evaluate_trust.py's volume-interval check reads -- omitted on
# the first pass, causing every volume_intervals.available to read 0. Fixed here.
[[ -f "$DIR/m1_calibration.json" ]] || fail "missing $DIR/m1_calibration.json (T7.7/T7.8 output) -- run calibrate.py first"
CUDA_VISIBLE_DEVICES=0 $PY scripts/extract_core_features.py --config artifacts/private/stage5_configs/M1.yaml \
  --checkpoint outputs/M1/checkpoints/best.pt --split val --mc-samples 10 \
  --calibration "$DIR/m1_calibration.json" \
  --output "$DIR/m1_val_features_mc10.json" 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "extract_core_features.py (M1, mc-samples=10)"

log "=== evaluate_trust.py: failure-detection AUROC, risk-coverage, random-referral control ==="
# extract_core_features.py writes the JSON evidence records to the literal --output path above
# (the .probe.npz sibling holds only raw feature/cohort/source arrays for the T5.11-style linear
# probe, not what evaluate_trust.py needs) -- reuse that same path here.
$PY scripts/evaluate_trust.py \
  --evaluation outputs/M1/val_cases.json \
  --features "$DIR/m1_val_features_mc10.json" \
  --output "$DIR/m1_trust_evaluation.json" \
  --case-output "$DIR/m1_trust_case_output.json" \
  --split val 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fail "evaluate_trust.py"

log "=== M5 trust pipeline complete ==="
echo "M5_TRUST_EXIT=0" | tee -a "$LOG"
