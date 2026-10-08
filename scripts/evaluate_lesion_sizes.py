#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np

from tumortrust_vlm.evaluation.metrics import lesion_detection, region_mask
from tumortrust_vlm.utils import atomic_json_dump


def summarize(cases: list[dict]) -> dict:
    metric_names = sorted(
        {
            key
            for case in cases
            for key, value in case.items()
            if key not in {"subject_id", "cohort"}
            and isinstance(value, (int, float))
            and np.isfinite(value)
        }
    )
    result = {
        "subjects": len(cases),
        "metrics": {
            key: float(np.mean([case[key] for case in cases if np.isfinite(case.get(key, np.nan))]))
            for key in metric_names
        },
    }
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_cohort[case["cohort"]].append(case)
    result["by_cohort"] = {
        cohort: {
            "subjects": len(records),
            "small_lesion_targets": int(sum(row["small_lesion_targets"] for row in records)),
            "small_lesion_tp": int(sum(row["small_lesion_tp"] for row in records)),
            "small_lesion_recall": (
                sum(row["small_lesion_tp"] for row in records)
                / max(1, sum(row["small_lesion_targets"] for row in records))
            ),
        }
        for cohort, records in sorted(by_cohort.items())
    }
    total_small = sum(row["small_lesion_targets"] for row in cases)
    result["pooled_small_lesion_targets"] = int(total_small)
    result["pooled_small_lesion_tp"] = int(sum(row["small_lesion_tp"] for row in cases))
    result["pooled_small_lesion_recall"] = (
        result["pooled_small_lesion_tp"] / total_small if total_small else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WT lesion detection by physical-size stratum from saved masks."
    )
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output")
    parser.add_argument("--minimum-mm3", type=float, default=100.0)
    parser.add_argument("--small-maximum-mm3", type=float, default=1000.0)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_final_test:
        raise SystemExit("Final test is locked")
    reference_root = Path(args.reference_root)
    prediction_root = Path(args.prediction_root)
    cases = []
    for reference_path in sorted(reference_root.glob("*/*.nii.gz")):
        relative = reference_path.relative_to(reference_root)
        prediction_path = prediction_root / relative
        if not prediction_path.is_file():
            raise SystemExit(f"Missing prediction for {relative}")
        reference_image = nib.load(reference_path)
        prediction_image = nib.load(prediction_path)
        spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        reference = region_mask(np.asarray(reference_image.dataobj), "WT")
        prediction = region_mask(np.asarray(prediction_image.dataobj), "WT")
        metrics = lesion_detection(
            prediction,
            reference,
            minimum_voxels=max(1, round(args.minimum_mm3 / voxel_volume)),
            small_lesion_maximum_voxels=max(1, round(args.small_maximum_mm3 / voxel_volume)),
        )
        cases.append(
            {
                "subject_id": reference_path.name.removesuffix(".nii.gz"),
                "cohort": relative.parts[0],
                **metrics,
            }
        )
    if not cases:
        raise SystemExit("No reference masks found")
    summary = {
        "split": args.split,
        "minimum_component_mm3": args.minimum_mm3,
        "small_lesion_range_mm3": [args.minimum_mm3, args.small_maximum_mm3],
        **summarize(cases),
    }
    atomic_json_dump(summary, args.output)
    if args.case_output:
        atomic_json_dump(cases, args.case_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
