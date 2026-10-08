#!/usr/bin/env bash
# Train two exact z-score seed replicas, then compare entropy, MC-dropout, and deep ensembles.
set -euo pipefail

cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY=(conda run --no-capture-output -n llava python)
seed0_config=artifacts/private/seed_configs/segmentation_zscore_seed0.yaml
seed1_config=artifacts/private/seed_configs/segmentation_zscore_seed1.yaml

train_seed() {
  local gpu="$1"
  local config="$2"
  local output="$3"
  local log="$4"
  if [[ ! -f "${output}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/train.py --config "$config" 2>&1 | tee "$log"
  fi
  if [[ ! -f "${output}/val_aggregate.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/evaluate.py \
      --config "${output}/resolved_config.json" \
      --checkpoint "${output}/checkpoints/best.pt" \
      --split val
  fi
}

train_seed 0 "$seed0_config" outputs/segmentation_zscore_seed0 logs_segmentation_zscore_seed0.log &
pid_seed0=$!
train_seed 1 "$seed1_config" outputs/segmentation_zscore_seed1 logs_segmentation_zscore_seed1.log &
pid_seed1=$!
wait "$pid_seed0"
wait "$pid_seed1"

extract_single() {
  local gpu="$1"
  local samples="$2"
  local slug="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/extract_core_features.py \
    --config outputs/D3_zscore/resolved_config.json \
    --checkpoint outputs/D3_zscore/checkpoints/best.pt \
    --checkpoint-config artifacts/private/ablation_configs_seed2/D3_zscore.yaml \
    --split val \
    --mc-samples "$samples" \
    --output "artifacts/private/trust/${slug}_features.json"
  "${PY[@]}" scripts/summarize_uncertainty_baseline.py \
    --method "$slug" \
    --features "artifacts/private/trust/${slug}_features.json" \
    --reference-cases outputs/D3_zscore/val_cases.json \
    --output "artifacts/development_uncertainty_${slug}.json"
}

extract_single 0 1 entropy &
pid_entropy=$!
extract_single 1 10 mc_dropout10 &
pid_mc=$!
wait "$pid_entropy"
wait "$pid_mc"

CUDA_VISIBLE_DEVICES=0 "${PY[@]}" scripts/extract_core_features.py \
  --config outputs/D3_zscore/resolved_config.json \
  --checkpoint \
    outputs/segmentation_zscore_seed0/checkpoints/best.pt \
    outputs/segmentation_zscore_seed1/checkpoints/best.pt \
    outputs/D3_zscore/checkpoints/best.pt \
  --checkpoint-config \
    outputs/segmentation_zscore_seed0/resolved_config.json \
    outputs/segmentation_zscore_seed1/resolved_config.json \
    artifacts/private/ablation_configs_seed2/D3_zscore.yaml \
  --split val \
  --mc-samples 1 \
  --output artifacts/private/trust/deep_ensemble3_features.json
"${PY[@]}" scripts/summarize_uncertainty_baseline.py \
  --method deep_ensemble3 \
  --features artifacts/private/trust/deep_ensemble3_features.json \
  --reference-cases outputs/D3_zscore/val_cases.json \
  --output artifacts/development_uncertainty_deep_ensemble3.json

printf 'MATCHED_UNCERTAINTY_IID_EXIT=0 %s\n' "$(date --iso-8601=seconds)"
