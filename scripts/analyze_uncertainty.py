#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from tumortrust_vlm.evaluation.trust import (
    failure_detection,
    random_referral_control,
    selective_risk_at_coverage,
)
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--shift-features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluation = {record["subject_id"]: record for record in json.loads(Path(args.evaluation).read_text())}
    features = {record["subject_id"]: record for record in json.loads(Path(args.features).read_text())}
    common = sorted(set(evaluation) & set(features))
    error = np.asarray([1 - evaluation[subject_id]["macro_dice"] for subject_id in common])
    segmentation_uncertainty = np.asarray(
        [features[subject_id]["evidence"]["segmentation_uncertainty"] for subject_id in common]
    )
    classification_uncertainty = np.asarray(
        [features[subject_id]["evidence"]["classification_uncertainty"] for subject_id in common]
    )
    result = {
        "subjects": len(common),
        **failure_detection(segmentation_uncertainty, error, threshold=0.20),
        "segmentation_risk_at_80pct": selective_risk_at_coverage(segmentation_uncertainty, error, 0.8),
        "classification_uncertainty_mean": float(classification_uncertainty.mean()),
        "random_referral_control": random_referral_control(
            segmentation_uncertainty, error, 0.8
        ),
    }
    if args.shift_features:
        shift = json.loads(Path(args.shift_features).read_text())
        iid_values = [record["evidence"]["segmentation_uncertainty"] for record in features.values()]
        shift_values = [record["evidence"]["segmentation_uncertainty"] for record in shift]
        labels = np.asarray([0] * len(iid_values) + [1] * len(shift_values))
        scores = np.asarray(iid_values + shift_values)
        result["shift_detection_auroc"] = float(roc_auc_score(labels, scores))
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
