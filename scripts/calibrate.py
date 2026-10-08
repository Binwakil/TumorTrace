#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import softmax

from tumortrust_vlm.evaluation.metrics import classification_metrics
from tumortrust_vlm.models.calibration import ConformalResidualInterval, TemperatureScaler
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.validation_predictions).read_text(encoding="utf-8"))
    probabilities = np.asarray([case["class_probability"] for case in cases], dtype=float)
    logits = np.log(np.clip(probabilities, 1e-8, 1))
    labels = np.asarray([case["class_target"] for case in cases], dtype=int)
    scaler = TemperatureScaler().fit(logits, labels)
    calibrated = softmax(scaler.transform(logits), axis=1)
    calibration = {
        "temperature": scaler.temperature,
        "classification_before": classification_metrics(probabilities, labels),
        "classification_after": classification_metrics(calibrated, labels),
        "volume_intervals": {},
    }
    for region in ("WT", "TC", "ET", "SNFH"):
        predicted = np.asarray([case[f"predicted_volume_ml_{region}"] for case in cases])
        target = np.asarray([case[f"target_volume_ml_{region}"] for case in cases])
        intervals = {}
        for coverage in (0.90, 0.95):
            model = ConformalResidualInterval(coverage).fit(predicted, target)
            low, high = model.predict(predicted)
            empirical = float(np.mean((target >= low) & (target <= high)))
            intervals[str(coverage)] = {
                "residual_quantile_ml": model.quantile,
                "empirical_validation_coverage": empirical,
                "mean_width_ml": float(np.mean(high - low)),
            }
        calibration["volume_intervals"][region] = intervals
    uncertainty = -np.sum(calibrated * np.log(np.clip(calibrated, 1e-8, 1)), axis=1)
    threshold = float(np.quantile(uncertainty, 0.80))
    calibration["classification_referral"] = {
        "target_coverage": 0.80,
        "uncertainty_threshold": threshold,
        "empirical_coverage": float(np.mean(uncertainty <= threshold)),
    }
    if args.features:
        features = {
            record["subject_id"]: record
            for record in json.loads(Path(args.features).read_text(encoding="utf-8"))
        }
        common = [case for case in cases if case["subject_id"] in features]
        segmentation_uncertainty = np.asarray(
            [
                features[case["subject_id"]]["evidence"]["segmentation_uncertainty"]
                for case in common
            ]
        )
        segmentation_threshold = float(np.quantile(segmentation_uncertainty, 0.80))
        calibration["segmentation_referral"] = {
            "target_coverage": 0.80,
            "uncertainty_threshold": segmentation_threshold,
            "empirical_coverage": float(
                np.mean(segmentation_uncertainty <= segmentation_threshold)
            ),
            "subjects": len(common),
        }
    atomic_json_dump(calibration, args.output)
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
