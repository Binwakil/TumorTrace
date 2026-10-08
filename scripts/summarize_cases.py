#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.metrics import (
    aggregate_case_metrics,
    classification_metrics,
)
from tumortrust_vlm.evaluation.stratification import stratified_case_metrics
from tumortrust_vlm.evaluation.volumetry import aggregate_volume_metrics
from tumortrust_vlm.utils import atomic_json_dump


def summarize(cases: list[dict], ece_bins: int = 15) -> dict:
    if not cases:
        raise ValueError("Cannot summarize an empty case file")
    regions = tuple(region for region in ("WT", "TC", "ET", "SNFH") if f"dice_{region}" in cases[0])
    result = aggregate_case_metrics(cases)
    result["subjects"] = len(cases)
    if regions:
        result.update(aggregate_volume_metrics(cases, regions=regions))
        result["stratified"] = stratified_case_metrics(cases)
    classification_cases = [
        case for case in cases if "class_probability" in case and "class_target" in case
    ]
    result["classification_evaluated"] = bool(classification_cases)
    if classification_cases:
        result.update(
            classification_metrics(
                np.asarray(
                    [case["class_probability"] for case in classification_cases],
                    dtype=float,
                ),
                np.asarray([case["class_target"] for case in classification_cases], dtype=int),
                ece_bins=ece_bins,
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    result = summarize(cases, args.ece_bins)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
