#!/usr/bin/env python
"""Fit the retained evidence calibration using development validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.metrics import classification_metrics
from tumortrust_vlm.models.calibration import ConformalResidualInterval, TemperatureScaler
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", default="outputs/C0/val_cases.json")
    parser.add_argument("--segmentation", default="outputs/D3_zscore/val_cases.json")
    parser.add_argument("--uncertainty", default="artifacts/development_uncertainty_selection.json")
    parser.add_argument("--output", default="artifacts/development_final_evidence_calibration.json")
    args = parser.parse_args()
    classification_path = ROOT / args.classification
    segmentation_path = ROOT / args.segmentation
    uncertainty_path = ROOT / args.uncertainty
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    if len(classification) != 248 or len(segmentation) != 248:
        raise ValueError("Final calibration requires the matched 248-case development validation")
    if not uncertainty.get("final_decision_available"):
        raise ValueError("Uncertainty/referral decision is not final")

    probabilities = np.asarray([row["class_probability"] for row in classification], float)
    labels = np.asarray([row["class_target"] for row in classification], int)
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    scaler = TemperatureScaler().fit(logits, labels)
    calibrated = np.exp(scaler.transform(logits))
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    result = {
        "scope": "development_validation_only",
        "subjects": 248,
        "temperature": float(scaler.temperature),
        "classification_before": classification_metrics(probabilities, labels),
        "classification_after": classification_metrics(calibrated, labels),
        "classification_limit": "source_limited_auxiliary_due_shortcut_gate_failure",
        "classification_referral": {
            "enabled": False,
            "reason": "shortcut_gate_failure",
        },
        "volume_intervals": {},
        "inputs": {
            args.classification: sha256_file(classification_path),
            args.segmentation: sha256_file(segmentation_path),
            args.uncertainty: sha256_file(uncertainty_path),
        },
        "locked_final_test_opened": False,
    }
    for region in ("WT", "TC", "ET", "SNFH"):
        predicted = np.asarray(
            [row[f"predicted_volume_ml_{region}"] for row in segmentation], float
        )
        target = np.asarray([row[f"target_volume_ml_{region}"] for row in segmentation], float)
        result["volume_intervals"][region] = {}
        for coverage in (0.90, 0.95):
            model = ConformalResidualInterval(coverage).fit(predicted, target)
            low, high = model.predict(predicted)
            result["volume_intervals"][region][str(coverage)] = {
                "residual_quantile_ml": float(model.quantile),
                "empirical_validation_coverage": float(np.mean((target >= low) & (target <= high))),
                "mean_width_ml": float(np.mean(high - low)),
            }

    retained = bool(uncertainty.get("retain_failure_detection_and_referral"))
    result["segmentation_referral"] = {"enabled": retained}
    if retained:
        threshold = uncertainty["development_referral_threshold"]
        result["segmentation_referral"].update(
            {
                "signal": threshold["signal"],
                "uncertainty_threshold": threshold["uncertainty_threshold"],
                "target_coverage": threshold["coverage"],
                "comparison": threshold["comparison"],
            }
        )
    else:
        result["segmentation_referral"]["reason"] = "development_retention_gates_failed"
    atomic_json_dump(result, ROOT / args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
