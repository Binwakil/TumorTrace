#!/usr/bin/env bash
# Matched development-only 7B decoder comparison frozen in configs/vlm_decoder_comparison.yaml.
set -euo pipefail

cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY=(conda run --no-capture-output -n llava python)
TRAIN=artifacts/private/reporter_records_train.json
VAL=artifacts/private/reporter_records_val.json

run_condition() {
  local gpu="$1"
  local slug="$2"
  local model="$3"
  local mode="$4"
  local output="outputs/${slug}_${mode}"
  local log="logs_${slug}_${mode}.log"

  if [[ ! -f "${output}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/train_vicuna_reporter.py \
      --train-records "$TRAIN" \
      --val-records "$VAL" \
      --output "$output" \
      --model "$model" \
      --mode "$mode" \
      --epochs 3 \
      --batch-size 1 \
      --accumulation-steps 8 \
      --maximum-length 256 \
      --learning-rate 0.0002 \
      --seed 20260810 \
      --protocol configs/vlm_decoder_comparison.yaml 2>&1 | tee "$log"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/evaluate_vicuna_reporter.py \
    --records "$VAL" \
    --checkpoint "${output}/best.pt" \
    --output "${output}/val_predictions_unconstrained.json" \
    --batch-size 4 \
    --maximum-new-tokens 256
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/evaluate_vicuna_reporter.py \
    --records "$VAL" \
    --checkpoint "${output}/best.pt" \
    --output "${output}/val_predictions_constrained.json" \
    --batch-size 4 \
    --maximum-new-tokens 256 \
    --schema-constrained
  CUDA_VISIBLE_DEVICES="$gpu" "${PY[@]}" scripts/probe_prefix_reporter_inputs.py \
    --records "$VAL" \
    --checkpoint "${output}/best.pt" \
    --output "${output}/val_input_probes.json" \
    --batch-size 4 \
    --maximum-new-tokens 256
}

run_model() {
  local gpu="$1"
  local slug="$2"
  local model="$3"
  local master_log="logs_${slug}_matrix.log"
  {
    printf 'START %s %s\n' "$slug" "$(date --iso-8601=seconds)"
    for mode in text_only evidence_only full; do
      run_condition "$gpu" "$slug" "$model" "$mode"
    done
    printf 'COMPLETE %s %s\n' "$slug" "$(date --iso-8601=seconds)"
  } 2>&1 | tee "$master_log"
}

run_model 0 llava_v15_vicuna_7b ../shared/models/llava-v1.5-7b &
pid_vicuna=$!
run_model 1 llava_med_v15_mistral_7b ../shared/models/llava-med-v1.5-mistral-7b &
pid_med=$!
wait "$pid_vicuna"
wait "$pid_med"
printf 'LLAVA_DECODER_COMPARISON_EXIT=0 %s\n' "$(date --iso-8601=seconds)"
