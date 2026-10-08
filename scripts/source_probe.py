#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tumortrust_vlm.utils import atomic_json_dump


def probe_target(features: np.ndarray, target: np.ndarray, name: str) -> dict:
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) < 2:
        return {
            f"{name}_balanced_accuracy": None,
            f"{name}_status": "infeasible_single_class",
            f"{name}_classes": len(classes),
        }
    folds = min(5, int(counts.min()))
    if folds < 2:
        return {
            f"{name}_balanced_accuracy": None,
            f"{name}_status": "infeasible_insufficient_class_count",
            f"{name}_classes": len(classes),
        }
    splitter = StratifiedKFold(folds, shuffle=True, random_state=20260810)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    prediction = cross_val_predict(model, features, target, cv=splitter)
    return {
        f"{name}_balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        f"{name}_status": "complete",
        f"{name}_classes": len(classes),
        f"{name}_cv_folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", required=True, help="NPZ with features [N,D], source [N], cohort [N]"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = np.load(args.features)
    features = payload["features"]
    source = payload["source"]
    cohort = payload["cohort"]
    result = {
        "subjects": len(features),
        **probe_target(features, source, "source"),
        **probe_target(features, cohort, "cohort"),
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
