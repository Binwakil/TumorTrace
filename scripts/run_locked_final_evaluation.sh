#!/usr/bin/env bash
# Execute the frozen 320-case final protocol exactly once. Never run before protocol freeze.
set -euo pipefail

cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY=(conda run --no-capture-output -n llava python)
REPORT_PY=artifacts/private/envs/report-metrics/bin/python
PROTOCOL=artifacts/final_protocol_lock.json
FINAL_ROOT=artifacts/locked_final
PRIVATE_ROOT=artifacts/private/locked_final
NNUNET_MODEL_ROOT=artifacts/private/nnunet_results
NNUNET_TEST_ROOT=artifacts/private/nnunet_final_test/Dataset501_TumorTrustFinalTest
NNUNET_PREDICTIONS="$PRIVATE_ROOT/nnunet_predictions"
completed=0

on_exit() {
  local status=$?
  if [[ $status -ne 0 && $completed -eq 0 ]]; then
    "${PY[@]}" scripts/manage_final_execution.py fail \
      --protocol "$PROTOCOL" --message "orchestrator_exit_${status}" || true
  fi
}
trap on_exit EXIT

"${PY[@]}" scripts/manage_final_execution.py begin --protocol "$PROTOCOL"
mkdir -p "$FINAL_ROOT" "$PRIVATE_ROOT" "$NNUNET_PREDICTIONS"

"${PY[@]}" scripts/export_nnunet_final_test.py \
  --output "$NNUNET_TEST_ROOT" \
  --mapping artifacts/private/nnunet_final_test_mapping.json \
  --unlock-final-test

nnUNet_raw=artifacts/private/nnunet_raw \
nnUNet_preprocessed=artifacts/private/nnunet_preprocessed \
nnUNet_results="$NNUNET_MODEL_ROOT" \
conda run --no-capture-output -n llava nnUNetv2_predict \
  -i "$NNUNET_TEST_ROOT/imagesTs" \
  -o "$NNUNET_PREDICTIONS" \
  -d 501 -c 3d_fullres -f 0 \
  -tr nnUNetTrainer -p nnUNetResEncUNetMPlans -chk checkpoint_final.pth

"${PY[@]}" scripts/evaluate_nnunet.py \
  --predictions "$NNUNET_PREDICTIONS" \
  --raw-dataset "$NNUNET_TEST_ROOT" \
  --mapping artifacts/private/nnunet_final_test_mapping.json \
  --split test --unlock-final-test \
  --case-output "$PRIVATE_ROOT/nnunet_cases.json" \
  --official-output "$PRIVATE_ROOT/nnunet_official.json" \
  --summary-output "$FINAL_ROOT/nnunet_summary.json"

CUDA_VISIBLE_DEVICES=0 "${PY[@]}" scripts/evaluate.py \
  --config outputs/D3_zscore/resolved_config.json \
  --checkpoint outputs/D3_zscore/checkpoints/best.pt \
  --split test --unlock-final-test \
  --case-output "$PRIVATE_ROOT/d3_cases.json" &
pid_d3=$!

CUDA_VISIBLE_DEVICES=1 "${PY[@]}" scripts/evaluate.py \
  --config outputs/C0/resolved_config.json \
  --checkpoint outputs/C0/checkpoints/best.pt \
  --split test --unlock-final-test \
  --case-output "$PRIVATE_ROOT/c0_cases.json" &
pid_c0=$!
d3_status=0
c0_status=0
wait "$pid_d3" || d3_status=$?
wait "$pid_c0" || c0_status=$?
if [[ $d3_status -ne 0 || $c0_status -ne 0 ]]; then
  printf 'Locked D3/C0 evaluation failed: d3=%s c0=%s\n' "$d3_status" "$c0_status" >&2
  exit 3
fi

selection=artifacts/development_uncertainty_selection.json
calibration=artifacts/development_final_evidence_calibration.json
retained="$(jq -r '.retain_failure_detection_and_referral' "$selection")"
method="$(jq -r '.selected_candidate.method' "$selection")"
checkpoints=(outputs/D3_zscore/checkpoints/best.pt)
checkpoint_configs=(artifacts/private/ablation_configs_seed2/D3_zscore.yaml)
mc_samples=1
disable_referral=(--disable-referral)
if [[ "$retained" == "true" ]]; then
  disable_referral=()
  case "$method" in
    entropy)
      ;;
    mc_dropout10)
      mc_samples=10
      ;;
    deep_ensemble3)
      checkpoints=(
        outputs/segmentation_zscore_seed0/checkpoints/best.pt
        outputs/segmentation_zscore_seed1/checkpoints/best.pt
        outputs/D3_zscore/checkpoints/best.pt
      )
      checkpoint_configs=(
        outputs/segmentation_zscore_seed0/resolved_config.json
        outputs/segmentation_zscore_seed1/resolved_config.json
        artifacts/private/ablation_configs_seed2/D3_zscore.yaml
      )
      ;;
    *)
      printf 'Unsupported frozen uncertainty method: %s\n' "$method" >&2
      exit 2
      ;;
  esac
fi

CUDA_VISIBLE_DEVICES=0 "${PY[@]}" scripts/extract_core_features.py \
  --config outputs/D3_zscore/resolved_config.json \
  --checkpoint outputs/D3_zscore/checkpoints/best.pt \
  --checkpoint-config artifacts/private/ablation_configs_seed2/D3_zscore.yaml \
  --split test --unlock-final-test \
  --mc-samples 1 \
  --calibration "$calibration" \
  --output "$PRIVATE_ROOT/d3_structured_evidence.json"

uncertainty_features="$PRIVATE_ROOT/d3_structured_evidence.json"
if [[ "$retained" == "true" && "$method" != "entropy" ]]; then
  uncertainty_features="$PRIVATE_ROOT/selected_uncertainty_features.json"
  CUDA_VISIBLE_DEVICES=0 "${PY[@]}" scripts/extract_core_features.py \
    --config outputs/D3_zscore/resolved_config.json \
    --checkpoint "${checkpoints[@]}" \
    --checkpoint-config "${checkpoint_configs[@]}" \
    --split test --unlock-final-test \
    --mc-samples "$mc_samples" \
    --output "$uncertainty_features"
fi

"${PY[@]}" scripts/apply_selected_referral.py \
  --segmentation "$PRIVATE_ROOT/d3_structured_evidence.json" \
  --uncertainty-features "$uncertainty_features" \
  --selection "$selection" \
  --output "$PRIVATE_ROOT/structured_evidence_segmentation.json" \
  --unlock-final-test

"${PY[@]}" scripts/merge_final_classification_evidence.py \
  --segmentation "$PRIVATE_ROOT/structured_evidence_segmentation.json" \
  --classification-cases "$PRIVATE_ROOT/c0_cases.json" \
  --calibration "$calibration" \
  --output "$PRIVATE_ROOT/structured_evidence.json" \
  "${disable_referral[@]}" --unlock-final-test

"${PY[@]}" scripts/build_report_records.py \
  --config outputs/D3_zscore/resolved_config.json \
  --features "$PRIVATE_ROOT/structured_evidence.json" \
  --split test --unlock-final-test \
  --output "$PRIVATE_ROOT/reporter_records.json"

"${PY[@]}" scripts/evaluate_deterministic_reports.py \
  --records "$PRIVATE_ROOT/reporter_records.json" \
  --output "$PRIVATE_ROOT/deterministic_report_predictions.json" \
  --unlock-final-test

CUDA_VISIBLE_DEVICES="" "$REPORT_PY" scripts/evaluate_clinical_text_metrics.py \
  --predictions "$PRIVATE_ROOT/deterministic_report_predictions.json" \
  --config configs/report_metrics.yaml \
  --split test --unlock-final-test \
  --output "$FINAL_ROOT/deterministic_clinical_summary.json" \
  --case-output "$PRIVATE_ROOT/deterministic_clinical_cases.json"

"${PY[@]}" scripts/summarize_locked_final.py \
  --root "$PRIVATE_ROOT" --protocol "$PROTOCOL" \
  --output artifacts/locked_final_summary.json \
  --bootstrap-samples 2000 --seed 20260822 --unlock-final-test

"${PY[@]}" scripts/manage_final_execution.py complete --protocol "$PROTOCOL" \
  --message "frozen_one_shot_protocol_completed"
completed=1
printf 'LOCKED_FINAL_EXIT=0 %s\n' "$(date --iso-8601=seconds)"
