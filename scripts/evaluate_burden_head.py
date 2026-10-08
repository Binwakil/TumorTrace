#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.train_burden_head import REGIONS, assemble, feature_provenance
from tumortrust_vlm.evaluation.volumetry import (
    aggregate_auxiliary_burden_metrics,
    aggregate_volume_metrics,
)
from tumortrust_vlm.models.burden import FrozenFeatureBurdenMLP
from tumortrust_vlm.utils import atomic_json_dump


def summarize(cases: list[dict]) -> dict:
    learned = aggregate_auxiliary_burden_metrics(cases, regions=REGIONS)
    deterministic = aggregate_volume_metrics(cases, regions=REGIONS)
    comparison = {}
    for region in REGIONS:
        learned_absolute = learned[f"burden_head_median_absolute_volume_error_ml_{region}"]
        deterministic_absolute = deterministic[f"median_absolute_volume_error_ml_{region}"]
        learned_relative = learned[f"burden_head_median_relative_volume_error_{region}"]
        deterministic_relative = deterministic[f"median_relative_volume_error_{region}"]
        comparison[region] = {
            "learned_median_absolute_error_ml": learned_absolute,
            "deterministic_median_absolute_error_ml": deterministic_absolute,
            "learned_minus_deterministic_absolute_error_ml": (
                learned_absolute - deterministic_absolute
            ),
            "learned_median_relative_error": learned_relative,
            "deterministic_median_relative_error": deterministic_relative,
            "learned_minus_deterministic_relative_error": (
                learned_relative - deterministic_relative
            ),
            "learned_ccc": learned[f"burden_head_ccc_{region}"],
            "deterministic_ccc": deterministic[f"ccc_{region}"],
        }
    return {"subjects": len(cases), "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen-feature burden MLP with deterministic mask volumetry."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_final_test:
        raise SystemExit("Final test is locked")
    subject_ids, inputs, _ = assemble(args.features, args.evaluation)
    evaluation = {
        row["subject_id"]: row
        for row in json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current_feature_provenance = feature_provenance(args.features)
    if current_feature_provenance != checkpoint.get("core_feature_provenance"):
        raise ValueError("Evaluation features were not produced by the training-time frozen core")
    model = FrozenFeatureBurdenMLP(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalized = (
        inputs - np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    ) / np.asarray(checkpoint["normalization_scale"], dtype=np.float32)
    with torch.no_grad():
        predicted = (
            torch.expm1(model(torch.from_numpy(normalized).to(device))).clamp_min(0).cpu().numpy()
        )
    cases = []
    for subject_id, prediction in zip(subject_ids, predicted, strict=True):
        source = evaluation[subject_id]
        case = {
            "subject_id": subject_id,
            "cohort": source["cohort"],
            "source_branch": source["source_branch"],
            "predicted_laterality": source["predicted_laterality"],
            "target_laterality": source["target_laterality"],
        }
        for index, region in enumerate(REGIONS):
            target = float(source[f"target_volume_ml_{region}"])
            learned = float(prediction[index])
            case[f"predicted_volume_ml_{region}"] = float(source[f"predicted_volume_ml_{region}"])
            case[f"target_volume_ml_{region}"] = target
            case[f"burden_head_predicted_volume_ml_{region}"] = learned
            case[f"burden_head_absolute_volume_error_ml_{region}"] = abs(learned - target)
            case[f"burden_head_relative_volume_error_{region}"] = (
                abs(learned - target) / target if target > 0 else float("nan")
            )
        cases.append(case)
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_cohort[case["cohort"]].append(case)
    summary = {
        "split": args.split,
        "checkpoint_epoch": checkpoint["epoch"],
        "input": "frozen_full_volume_global_3d_features",
        "mask_derived_volume_used_as_input": False,
        "overall": summarize(cases),
        "by_cohort": {cohort: summarize(records) for cohort, records in sorted(by_cohort.items())},
        "feature_inference_provenance": current_feature_provenance,
    }
    atomic_json_dump(summary, args.output)
    if args.case_output:
        atomic_json_dump(cases, args.case_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
