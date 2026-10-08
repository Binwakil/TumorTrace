#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.inference import evidence_vector
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge one-shot D3 test evidence with auxiliary C0 class probabilities."
    )
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--classification-cases", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--disable-referral", action="store_true")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_final_test:
        raise SystemExit("Final reporter evidence merge is locked")

    segmentation_path = Path(args.segmentation)
    classification_path = Path(args.classification_cases)
    calibration_path = Path(args.calibration)
    segmentation_rows = json.loads(segmentation_path.read_text(encoding="utf-8"))
    classification_rows = json.loads(classification_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if len(segmentation_rows) != 320 or len(classification_rows) != 320:
        raise ValueError("Final evidence merge requires exactly 320 structured-test cases")
    classification = {row["subject_id"]: row for row in classification_rows}
    if len(classification) != 320:
        raise ValueError("Duplicate classification subjects in final evidence merge")
    if {row["subject_id"] for row in segmentation_rows} != set(classification):
        raise ValueError("Final segmentation/classification subject sets differ")

    output = []
    class_names = ("GLI", "MEN", "MET")
    temperature = float(calibration["temperature"])
    for source in segmentation_rows:
        record = copy.deepcopy(source)
        class_case = classification[record["subject_id"]]
        raw_probability = np.asarray(class_case["class_probability"], dtype=float)
        logits = np.log(np.clip(raw_probability, 1e-8, 1.0)) / temperature
        calibrated_probability = np.exp(logits - logits.max())
        calibrated_probability /= calibrated_probability.sum()
        probabilities = {
            name: float(calibrated_probability[index]) for index, name in enumerate(class_names)
        }
        evidence = record["evidence"]
        evidence["tumor_family_probabilities"] = probabilities
        evidence["predicted_family"] = max(probabilities, key=probabilities.get)
        evidence["classification_uncertainty"] = float(
            -np.sum(calibrated_probability * np.log(np.clip(calibrated_probability, 1e-8, 1.0)))
        )
        evidence["referral_reasons"] = [
            reason
            for reason in evidence.get("referral_reasons", [])
            if reason != "classification_uncertainty"
        ]
        evidence["unavailable_fields"] = sorted(
            set(evidence.get("unavailable_fields", [])) | {"tumor_family"}
        )
        if args.disable_referral:
            evidence["referral_reasons"] = []
            evidence["unavailable_fields"] = sorted(
                set(evidence.get("unavailable_fields", [])) | {"referral"}
            )
        evidence["referral"] = bool(evidence["referral_reasons"])
        record["evidence_vector"] = evidence_vector(evidence)
        record["inference_provenance"] = {
            "composition": "locked_test_d3_evidence_plus_c0_auxiliary_classification_v1",
            "segmentation": record.get("inference_provenance", {}),
            "classification_cases_sha256": sha256_file(classification_path),
            "calibration_sha256": sha256_file(calibration_path),
            "classification_limit": "source_limited_auxiliary_due_shortcut_gate_failure",
        }
        record["subject_provenance"] = {
            "feature_extraction_split": "test",
            "excluded_from_checkpoint_training": True,
            "segmentation_recipe_excluded_from_training": True,
            "classification_recipe_excluded_from_training": True,
        }
        output.append(record)

    atomic_json_dump(output, args.output)
    summary = {
        "subjects": len(output),
        "split": "test",
        "segmentation_sha256": sha256_file(segmentation_path),
        "classification_cases_sha256": sha256_file(classification_path),
        "calibration_sha256": sha256_file(calibration_path),
        "classification_temperature": temperature,
        "referral_disabled": args.disable_referral,
        "classification_limit": "source_limited_auxiliary_due_shortcut_gate_failure",
        "locked_test_opened": True,
    }
    atomic_json_dump(summary, Path(args.output).with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
