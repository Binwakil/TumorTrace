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

from tumortrust_vlm.data.constants import COHORTS
from tumortrust_vlm.evaluation.metrics import classification_metrics
from tumortrust_vlm.evaluation.trust import failure_detection, random_referral_control
from tumortrust_vlm.utils import atomic_json_dump

REGIONS = ("WT", "TC", "ET", "SNFH")


def evaluate_group(rows: list[dict], random_samples: int = 2000) -> dict:
    if not rows:
        raise ValueError("Trust evaluation group cannot be empty")
    error = np.asarray([1 - row["case"]["macro_dice"] for row in rows], dtype=float)
    uncertainty = np.asarray(
        [row["feature"]["evidence"]["segmentation_uncertainty"] for row in rows],
        dtype=float,
    )
    result = {
        "subjects": len(rows),
        "mean_segmentation_error": float(error.mean()),
        **failure_detection(uncertainty, error, threshold=0.20),
        "referral_at_80pct": random_referral_control(
            uncertainty,
            error,
            coverage=0.80,
            samples=random_samples,
        ),
        "volume_intervals": {},
    }
    for region in REGIONS:
        region_result = {}
        target = np.asarray(
            [row["case"][f"target_volume_ml_{region}"] for row in rows], dtype=float
        )
        for coverage, field in ((0.90, "interval_90_ml"), (0.95, "interval_95_ml")):
            intervals = [row["feature"]["evidence"]["volumes"][region].get(field) for row in rows]
            available = np.asarray([interval is not None for interval in intervals])
            if not available.any():
                region_result[str(coverage)] = {
                    "available": 0,
                    "coverage": None,
                    "mean_width_ml": None,
                }
                continue
            bounds = np.asarray([interval for interval in intervals if interval is not None])
            selected_target = target[available]
            region_result[str(coverage)] = {
                "available": int(available.sum()),
                "coverage": float(
                    np.mean((selected_target >= bounds[:, 0]) & (selected_target <= bounds[:, 1]))
                ),
                "mean_width_ml": float(np.mean(bounds[:, 1] - bounds[:, 0])),
            }
        result["volume_intervals"][region] = region_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated intervals and uncertainty referral by development stratum."
    )
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--random-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_final_test:
        raise SystemExit("Final test is locked")
    cases = {
        row["subject_id"]: row
        for row in json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    }
    features = {
        row["subject_id"]: row
        for row in json.loads(Path(args.features).read_text(encoding="utf-8"))
    }
    if set(cases) != set(features):
        raise SystemExit(
            f"Evaluation/features subject mismatch: {len(cases)} versus {len(features)}"
        )
    rows = [
        {"subject_id": subject_id, "case": cases[subject_id], "feature": features[subject_id]}
        for subject_id in sorted(cases)
    ]
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cohort[row["case"]["cohort"]].append(row)
        by_source[row["case"]["source_branch"]].append(row)
    probabilities = np.asarray(
        [
            [row["feature"]["evidence"]["tumor_family_probabilities"][name] for name in COHORTS]
            for row in rows
        ]
    )
    labels = np.asarray([row["case"]["class_target"] for row in rows], dtype=int)
    result = {
        "split": args.split,
        "evaluation_scope": (
            "calibration_fit_diagnostic_not_independent"
            if args.split == "val"
            else "locked_heldout_evaluation"
        ),
        "overall": evaluate_group(rows, args.random_samples),
        "classification": classification_metrics(probabilities, labels),
        "by_cohort": {
            name: evaluate_group(group, args.random_samples)
            for name, group in sorted(by_cohort.items())
        },
        "by_source": {
            name: evaluate_group(group, args.random_samples)
            for name, group in sorted(by_source.items())
        },
    }
    if args.case_output:
        case_rows = []
        for row in rows:
            case = row["case"]
            evidence = row["feature"]["evidence"]
            case_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "cohort": case["cohort"],
                    "source_branch": case["source_branch"],
                    "segmentation_error": 1 - case["macro_dice"],
                    "segmentation_uncertainty": evidence["segmentation_uncertainty"],
                    "referral": evidence["referral"],
                }
            )
        atomic_json_dump(case_rows, args.case_output)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
