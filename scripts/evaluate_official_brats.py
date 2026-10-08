#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_evaluation import config_path, evaluate_single_exam
from panoptica import Panoptica_Evaluator

from tumortrust_vlm.utils import atomic_json_dump

CONFIGS = {"GLI": "gli", "MEN": "MenPre", "MET": "mets"}


def aggregate_results(results: list[dict]) -> dict:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in results:
        for region, payload in record.items():
            if not isinstance(payload, dict):
                continue
            for metric, value in payload.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values[region][metric].append(float(value))
    return {
        region: {
            metric: sum(metric_values) / len(metric_values)
            for metric, metric_values in sorted(metrics.items())
            if metric_values
        }
        for region, metrics in sorted(values.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--cohort", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.export_root)
    predictions = root / "prediction" / args.cohort
    references = root / "reference" / args.cohort
    output = Path(args.output)
    if args.reuse_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        results = json.loads(output.read_text(encoding="utf-8"))
    else:
        evaluator = Panoptica_Evaluator.load_from_config(str(config_path(CONFIGS[args.cohort])))
        results = []
        paths = sorted(predictions.glob("*.nii.gz"))
        if args.limit is not None:
            paths = paths[: args.limit]
        for prediction in paths:
            reference = references / prediction.name
            if not reference.is_file():
                raise FileNotFoundError(reference)
            results.append(
                evaluate_single_exam(
                    prediction_filepath=str(prediction),
                    reference_filepath=str(reference),
                    subject_identifier=prediction.name.removesuffix(".nii.gz"),
                    evaluator=evaluator,
                )
            )
        atomic_json_dump(results, output)
    summary = {
        "cohort": args.cohort,
        "subjects": len(results),
        "official_package": "BraTS-evaluation",
        "metrics": aggregate_results(results),
    }
    summary_output = (
        Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    )
    atomic_json_dump(summary, summary_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
