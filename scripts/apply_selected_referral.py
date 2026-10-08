#!/usr/bin/env python
"""Attach the frozen trust signal to D3 evidence without replacing D3 segmentation geometry."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.inference import evidence_vector
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--uncertainty-features", required=True)
    parser.add_argument("--selection", default="artifacts/development_uncertainty_selection.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_final_test:
        raise SystemExit("Final referral composition is locked")

    segmentation_path = Path(args.segmentation)
    uncertainty_path = Path(args.uncertainty_features)
    selection_path = Path(args.selection)
    segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("final_decision_available"):
        raise ValueError("Development uncertainty decision is not final")
    if len(segmentation) != 320 or len(uncertainty) != 320:
        raise ValueError("Final referral composition requires exactly 320 cases")
    uncertainty_map = {row["subject_id"]: row for row in uncertainty}
    if len(uncertainty_map) != 320:
        raise ValueError("Duplicate uncertainty subject IDs")
    if {row["subject_id"] for row in segmentation} != set(uncertainty_map):
        raise ValueError("Segmentation and uncertainty subject sets differ")

    retained = bool(selection.get("retain_failure_detection_and_referral"))
    threshold = selection["development_referral_threshold"]
    uncertainty_field = threshold["uncertainty_field"]
    evidence_field = {
        "segmentation_predictive_entropy_p95": "segmentation_predictive_entropy_p95",
        "segmentation_mutual_information_p95": "segmentation_mutual_information_p95",
    }[uncertainty_field]
    output = []
    referred = 0
    for source in segmentation:
        record = copy.deepcopy(source)
        trust = uncertainty_map[record["subject_id"]]
        value = float(trust["uncertainty"][evidence_field])
        evidence = record["evidence"]
        reasons = [
            reason
            for reason in evidence.get("referral_reasons", [])
            if reason != "segmentation_uncertainty"
        ]
        evidence["segmentation_uncertainty"] = value
        if retained and value > float(threshold["uncertainty_threshold"]):
            reasons.append("segmentation_uncertainty")
            referred += 1
        evidence["referral_reasons"] = reasons
        evidence["referral"] = bool(reasons)
        if not retained:
            evidence["unavailable_fields"] = sorted(
                set(evidence.get("unavailable_fields", [])) | {"referral"}
            )
        record["evidence_vector"] = evidence_vector(evidence)
        record["inference_provenance"] = {
            "segmentation_geometry": record.get("inference_provenance", {}),
            "uncertainty_only": trust.get("inference_provenance", {}),
            "selection_sha256": sha256_file(selection_path),
            "geometry_policy": "frozen_d3_segmentation_not_replaced_by_uncertainty_ensemble",
        }
        output.append(record)

    atomic_json_dump(output, args.output)
    summary = {
        "subjects": len(output),
        "retained": retained,
        "selected_method": selection["selected_candidate"]["method"],
        "selected_signal": selection["selected_candidate"]["signal"],
        "threshold": threshold["uncertainty_threshold"] if retained else None,
        "referred": referred if retained else None,
        "segmentation_sha256": sha256_file(segmentation_path),
        "uncertainty_features_sha256": sha256_file(uncertainty_path),
        "selection_sha256": sha256_file(selection_path),
        "locked_test_opened": True,
    }
    atomic_json_dump(summary, Path(args.output).with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
