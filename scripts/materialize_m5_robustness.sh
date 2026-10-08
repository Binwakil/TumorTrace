#!/usr/bin/env bash
# Materialize inference-only M5 robustness configs matched to the frozen D3 z-score baseline.
set -euo pipefail
cd /data/Projects/llava-tumor-baseline/Brain-VLMs/TumorTrace
PY="conda run -n llava python"
BASE=outputs/D3_zscore/resolved_config.json
DIR=artifacts/private/m5_robustness/configs
mkdir -p "$DIR"

for modality in t1n t1c t2w t2f; do
  $PY scripts/materialize_matched_config.py \
    --base "$BASE" \
    --override "project.name=TumorTrust-missing-${modality}-development" \
    --override "project.output_dir=outputs/m5_missing_${modality}_d3" \
    --override "project.operation=evaluate" \
    --override "data.force_missing_modality=${modality}" \
    --allow-changed data.force_missing_modality \
    --output "$DIR/missing_${modality}.yaml"
done

for pair in noise=gaussian_noise intensity=intensity_scale bias_field=bias_field resolution=lower_resolution; do
  name=${pair%%=*}
  shift=${pair#*=}
  $PY scripts/materialize_matched_config.py \
    --base "$BASE" \
    --override "project.name=TumorTrust-shift-${name}-development" \
    --override "project.output_dir=outputs/m5_shift_${name}_d3" \
    --override "project.operation=evaluate" \
    --override "data.robustness_shift=${shift}" \
    --allow-changed data.robustness_shift \
    --output "$DIR/shift_${name}.yaml"
done
