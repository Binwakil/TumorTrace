#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def summarize_radeval_results(results: dict[str, list[float]]) -> dict:
    summary = {}
    for metric, values in results.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError(f"RadEval returned invalid per-case values for {metric}")
        if metric == "radcliq_v1":
            raw_mean = float(array.mean())
            summary["radcliq_v1_raw_mean"] = raw_mean
            summary["radcliq_v1_inverse_mean"] = 1.0 / raw_mean if raw_mean != 0 else None
        else:
            summary[metric] = float(array.mean())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pinned upstream clinical report metrics on private paired predictions."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--config", default="configs/report_metrics.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_final_test:
        raise SystemExit("Final report test metrics are locked")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    package_version = version(config["implementation"]["package"])
    if package_version != str(config["implementation"]["version"]):
        raise SystemExit(
            f"RadEval version mismatch: installed {package_version}, "
            f"required {config['implementation']['version']}"
        )
    rows = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("Prediction file is empty")
    subject_ids = [row["subject_id"] for row in rows]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Prediction file contains duplicate subject IDs")
    record_splits = {row.get("report_split") for row in rows if row.get("report_split")}
    if record_splits and record_splits != {args.split}:
        raise ValueError(f"Prediction/report split mismatch: {sorted(record_splits)}")
    references = [row["reference"] for row in rows]
    hypotheses = [row["prediction"] for row in rows]
    if any(not text.strip() for text in references + hypotheses):
        raise ValueError("Clinical text metrics require non-empty paired texts")
    import radeval

    # RadEval 2.2.1's vendored RadGraph uses a case-sensitive ``RadEval`` absolute import even
    # though the installed Linux package is named ``radeval``. Alias only the already imported,
    # pinned package at runtime; do not mutate the locked environment or vendor code.
    sys.modules.setdefault("RadEval", radeval)
    from radeval import RadEval

    metrics = list(config["metrics"])
    evaluator = RadEval(metrics=metrics, per_sample=True, show_progress=True)
    results = evaluator(refs=references, hyps=hypotheses)
    expected_cases = len(rows)
    for metric, values in results.items():
        if len(values) != expected_cases:
            raise ValueError(f"RadEval returned {len(values)} {metric} values for {expected_cases} cases")
    case_records = []
    for index, subject_id in enumerate(subject_ids):
        case_records.append(
            {
                "subject_id": subject_id,
                **{metric: float(values[index]) for metric, values in results.items()},
            }
        )
    summary = {
        "split": args.split,
        "subjects": expected_cases,
        "input_sha256": sha256_file(args.predictions),
        "config": args.config,
        "config_sha256": sha256_file(ROOT / args.config),
        "implementation": {**config["implementation"], "installed_version": package_version},
        "metrics_requested": metrics,
        "interpretation": config["interpretation"],
        "domain_warning": config["domain_warning"],
        **summarize_radeval_results(results),
    }
    atomic_json_dump(summary, args.output)
    if args.case_output:
        atomic_json_dump(case_records, args.case_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
