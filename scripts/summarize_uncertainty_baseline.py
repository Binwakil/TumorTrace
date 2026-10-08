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
from sklearn.metrics import roc_auc_score

from tumortrust_vlm.evaluation.metrics import dice_score, region_mask
from tumortrust_vlm.evaluation.trust import failure_detection, random_referral_control
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

REGIONS = ("WT", "TC", "ET", "SNFH")
UNCERTAINTY_FIELDS = {
    "predictive_entropy": "segmentation_predictive_entropy_p95",
    "mutual_information": "segmentation_mutual_information_p95",
}


def evaluate_group(rows: list[dict]) -> dict:
    error = np.asarray([1 - row["macro_dice"] for row in rows], dtype=float)
    result = {"subjects": len(rows), "macro_dice": float(1 - error.mean())}
    for label, field in UNCERTAINTY_FIELDS.items():
        uncertainty = np.asarray([row[field] for row in rows], dtype=float)
        result[label] = {
            **failure_detection(uncertainty, error, threshold=0.20),
            "referral_at_80pct": random_referral_control(
                uncertainty, error, coverage=0.80
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize matched entropy, MC-dropout, or deep-ensemble uncertainty."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--uncertainty-map-dir")
    parser.add_argument("--reference-cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--shift",
        action="append",
        default=[],
        metavar="NAME=FEATURES_JSON",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    feature_rows = json.loads(features_path.read_text(encoding="utf-8"))
    features = {row["subject_id"]: row for row in feature_rows}
    reference_path = Path(args.reference_cases)
    references = {
        row["subject_id"]: row
        for row in json.loads(reference_path.read_text(encoding="utf-8"))
    }
    if set(features) != set(references):
        raise ValueError("Uncertainty features and reference cases have different subjects")

    rows = []
    map_root = Path(args.uncertainty_map_dir) if args.uncertainty_map_dir else None
    for subject_id in sorted(features):
        feature = features[subject_id]
        segmentation = feature.get("segmentation_evaluation")
        if segmentation is None:
            if map_root is None:
                raise ValueError(
                    "Features lack segmentation_evaluation and no map directory was given"
                )
            map_path = map_root / f"{subject_id}.npz"
            with np.load(map_path) as payload:
                prediction = payload["prediction"]
                target = payload["target"]
            dice = {
                region: dice_score(
                    region_mask(prediction, region), region_mask(target, region)
                )
                for region in REGIONS
            }
            segmentation = {
                **{f"dice_{region}": value for region, value in dice.items()},
                "macro_dice": float(np.mean(list(dice.values()))),
            }
        uncertainty = feature["uncertainty"]
        reference = references[subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "cohort": reference["cohort"],
                "source_branch": reference["source_branch"],
                "macro_dice": float(segmentation["macro_dice"]),
                "reference_macro_dice": float(reference["macro_dice"]),
                **{
                    f"dice_{region}": float(segmentation[f"dice_{region}"])
                    for region in REGIONS
                },
                **{
                    field: float(uncertainty[field])
                    for field in UNCERTAINTY_FIELDS.values()
                },
            }
        )

    by_cohort: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort"]].append(row)
        by_source[row["source_branch"]].append(row)
    paired_delta = np.asarray(
        [row["macro_dice"] - row["reference_macro_dice"] for row in rows], dtype=float
    )
    result = {
        "method": args.method,
        "features_sha256": sha256_file(features_path),
        "reference_cases_sha256": sha256_file(reference_path),
        "uncertainty_map_dir": str(map_root) if map_root else None,
        "overall": evaluate_group(rows),
        "paired_macro_dice_delta_vs_single_d3": float(paired_delta.mean()),
        "by_cohort": {
            name: evaluate_group(group) for name, group in sorted(by_cohort.items())
        },
        "by_source": {
            name: evaluate_group(group) for name, group in sorted(by_source.items())
        },
        "shift_detection": {},
    }
    iid_values = {
        label: [row[field] for row in rows]
        for label, field in UNCERTAINTY_FIELDS.items()
    }
    for specification in args.shift:
        if "=" not in specification:
            raise ValueError(f"Invalid --shift value: {specification!r}")
        name, path_value = specification.split("=", 1)
        shift_path = Path(path_value)
        shift_rows = json.loads(shift_path.read_text(encoding="utf-8"))
        if len(shift_rows) != len(rows):
            raise ValueError(f"Shift {name} subject count does not match IID")
        shift_result = {
            "features_sha256": sha256_file(shift_path),
            "subjects": len(shift_rows),
        }
        for label, field in UNCERTAINTY_FIELDS.items():
            shifted = [record["uncertainty"][field] for record in shift_rows]
            labels = np.asarray([0] * len(rows) + [1] * len(shifted))
            scores = np.asarray(iid_values[label] + shifted)
            shift_result[f"{label}_auroc"] = float(roc_auc_score(labels, scores))
        result["shift_detection"][name] = shift_result
    atomic_json_dump(result, args.output)
    case_output = Path(args.output).with_name(Path(args.output).stem + "_cases.json")
    atomic_json_dump(rows, case_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
