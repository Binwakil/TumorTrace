#!/usr/bin/env bash
# Secondary clinical-text metrics on prespecified development reporting conditions.
set -euo pipefail

cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY=artifacts/private/envs/report-metrics/bin/python

run_metrics() {
  local slug="$1"
  local predictions="$2"
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/evaluate_clinical_text_metrics.py \
    --predictions "$predictions" \
    --split val \
    --output "artifacts/development_clinical_text_metrics_${slug}.json" \
    --case-output "artifacts/private/clinical_text_metrics_${slug}_cases.json"
}

run_metrics deterministic outputs/reporter_deterministic_evidence/val_predictions.json
run_metrics compact_evidence outputs/reporter_evidence_only_compact/val_predictions_unconstrained.json
run_metrics llava_v15_vicuna_7b_full \
  outputs/llava_v15_vicuna_7b_full/val_predictions_unconstrained.json
run_metrics llava_med_v15_mistral_7b_evidence \
  outputs/llava_med_v15_mistral_7b_evidence_only/val_predictions_unconstrained.json
run_metrics llava_med_v15_mistral_7b_full \
  outputs/llava_med_v15_mistral_7b_full/val_predictions_unconstrained.json

printf 'DEVELOPMENT_CLINICAL_METRICS_EXIT=0 %s\n' "$(date --iso-8601=seconds)"
