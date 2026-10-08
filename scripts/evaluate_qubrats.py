#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.data.constants import REGION_LABELS
from tumortrust_vlm.evaluation.trust import qu_brats_score
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--uncertainty",
        choices=("predictive_entropy", "mutual_information"),
        default="mutual_information",
    )
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--maximum-uncertainty", type=float, default=math.log(4))
    args = parser.parse_args()
    cases = []
    for path in sorted(Path(args.maps).glob("*.npz")):
        with np.load(path) as payload:
            prediction = payload["prediction"]
            target = payload["target"]
            uncertainty = payload[args.uncertainty].astype(np.float32)
            brain = payload["brain_mask"]
        normalized = np.clip(uncertainty / args.maximum_uncertainty, 0.0, 1.0)
        result = {"subject_id": path.stem, "regions": {}}
        for region in ("WT", "TC", "ET"):
            labels = REGION_LABELS[region]
            metrics = qu_brats_score(
                np.isin(prediction, labels),
                np.isin(target, labels),
                normalized,
                brain,
                points=args.points,
            )
            result["regions"][region] = {
                key: value
                for key, value in metrics.items()
                if key not in {"thresholds", "dice", "ftp", "ftn"}
            }
        result["mean_score"] = float(
            np.mean([payload["score"] for payload in result["regions"].values()])
        )
        cases.append(result)
    summary = {
        "subjects": len(cases),
        "uncertainty": args.uncertainty,
        "maximum_uncertainty": args.maximum_uncertainty,
        "mean_score": float(np.mean([case["mean_score"] for case in cases])),
        "regions": {
            region: float(np.mean([case["regions"][region]["score"] for case in cases]))
            for region in ("WT", "TC", "ET")
        },
        "definition": "QU-BraTS Dice/FTP/FTN threshold-AUC score",
    }
    atomic_json_dump(cases, args.output)
    atomic_json_dump(summary, Path(args.output).with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
