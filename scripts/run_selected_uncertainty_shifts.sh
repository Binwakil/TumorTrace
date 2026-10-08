#!/usr/bin/env bash
# Evaluate the prespecified IID-selected uncertainty signal under all locked development shifts.
set -euo pipefail

cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY=(conda run --no-capture-output -n llava python)
protocol=configs/uncertainty_referral_closure.yaml
selection=artifacts/development_uncertainty_selection.json

"${PY[@]}" scripts/select_uncertainty_baseline.py \
  --config "$protocol" \
  --output "$selection"
method="$(jq -r '.selected_candidate.method' "$selection")"

checkpoints=()
checkpoint_configs=()
mc_samples=1
case "$method" in
  entropy)
    checkpoints=(outputs/D3_zscore/checkpoints/best.pt)
    checkpoint_configs=(artifacts/private/ablation_configs_seed2/D3_zscore.yaml)
    ;;
  mc_dropout10)
    checkpoints=(outputs/D3_zscore/checkpoints/best.pt)
    checkpoint_configs=(artifacts/private/ablation_configs_seed2/D3_zscore.yaml)
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
    printf 'Unsupported selected uncertainty method: %s\n' "$method" >&2
    exit 2
    ;;
esac

extract_shift() {
  local gpu="$1"
  local shift="$2"
  local config="artifacts/private/m5_robustness/configs/shift_${shift}.yaml"
  local output="artifacts/private/trust/${method}_shift_${shift}_features.json"
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/extract_core_features.py \
    --config "$config" \
    --checkpoint "${checkpoints[@]}" \
    --checkpoint-config "${checkpoint_configs[@]}" \
    --split val \
    --mc-samples "$mc_samples" \
    --output "$output"
}

extract_shift 0 noise &
pid_noise=$!
extract_shift 1 intensity &
pid_intensity=$!
wait "$pid_noise"
wait "$pid_intensity"

extract_shift 0 bias_field &
pid_bias=$!
extract_shift 1 resolution &
pid_resolution=$!
wait "$pid_bias"
wait "$pid_resolution"

shift_summary="artifacts/development_uncertainty_${method}_with_shifts.json"
"${PY[@]}" scripts/summarize_uncertainty_baseline.py \
  --method "$method" \
  --features "artifacts/private/trust/${method}_features.json" \
  --reference-cases outputs/D3_zscore/val_cases.json \
  --shift "noise=artifacts/private/trust/${method}_shift_noise_features.json" \
  --shift "intensity=artifacts/private/trust/${method}_shift_intensity_features.json" \
  --shift "bias_field=artifacts/private/trust/${method}_shift_bias_field_features.json" \
  --shift "resolution=artifacts/private/trust/${method}_shift_resolution_features.json" \
  --output "$shift_summary"

"${PY[@]}" scripts/select_uncertainty_baseline.py \
  --config "$protocol" \
  --shift-summary "$shift_summary" \
  --output "$selection"

printf 'MATCHED_UNCERTAINTY_SHIFT_EXIT=0 %s\n' "$(date --iso-8601=seconds)"
