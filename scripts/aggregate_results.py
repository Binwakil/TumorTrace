#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.statistics import bootstrap_ci, paired_bootstrap_difference
from tumortrust_vlm.utils import atomic_json_dump

TARGETS = {
    "macro_dice_improvement": 0.01,
    "balanced_accuracy": 0.86,
    "macro_f1": 0.85,
    "macro_auroc": 0.95,
    "ece": 0.05,
    "median_relative_volume_error_WT": 0.10,
    "median_relative_volume_error_TC": 0.15,
    "median_relative_volume_error_ET": 0.15,
    "median_relative_volume_error_SNFH": 0.15,
}


def load_cases(paths: list[str]) -> list[dict[str, dict]]:
    runs = []
    for path in paths:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        runs.append({record["subject_id"]: record for record in records})
    return runs


def summarize_runs(runs: list[dict[str, dict]], bootstrap_samples: int) -> dict:
    common = sorted(set.intersection(*(set(run) for run in runs)))
    if not common:
        raise ValueError("Runs have no paired subjects")
    numeric_keys = sorted(
        set.intersection(
            *(
                {
                    key
                    for subject_id in common
                    for key, value in run[subject_id].items()
                    if isinstance(value, (int, float)) and np.isfinite(value)
                }
                for run in runs
            ),
            *(
                {
                    key
                    for key, value in run[subject_id].items()
                    if isinstance(value, (int, float)) and np.isfinite(value)
                }
                for run in runs
                for subject_id in common
            ),
        )
    )
    summary = {"runs": len(runs), "paired_subjects": len(common), "metrics": {}}
    for key in numeric_keys:
        per_seed = []
        pooled_subject = []
        for run in runs:
            values = np.asarray([run[subject_id][key] for subject_id in common], dtype=float)
            per_seed.append(float(values.mean()))
            pooled_subject.append(values)
        subject_means = np.stack(pooled_subject).mean(0)
        low, high = bootstrap_ci(subject_means, samples=bootstrap_samples)
        summary["metrics"][key] = {
            "mean": float(subject_means.mean()),
            "ci_low": low,
            "ci_high": high,
            "seed_mean": float(np.mean(per_seed)),
            "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
            "seed_means": per_seed,
        }
    cohort_subjects: dict[str, list[str]] = defaultdict(list)
    for subject_id in common:
        cohort_subjects[runs[0][subject_id]["cohort"]].append(subject_id)
    cohort_summary = {}
    for cohort, subject_ids in sorted(cohort_subjects.items()):
        subject_values = np.asarray(
            [np.mean([run[subject_id]["macro_dice"] for run in runs]) for subject_id in subject_ids]
        )
        low, high = bootstrap_ci(subject_values, samples=bootstrap_samples)
        seed_means = [
            float(np.mean([run[subject_id]["macro_dice"] for subject_id in subject_ids]))
            for run in runs
        ]
        cohort_summary[cohort] = {
            "subjects": len(subject_ids),
            "mean": float(subject_values.mean()),
            "ci_low": low,
            "ci_high": high,
            "seed_sd": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "seed_means": seed_means,
        }
    cohort_means = [payload["mean"] for payload in cohort_summary.values()]
    summary["macro_dice_by_cohort"] = cohort_summary
    summary["macro_dice_worst_cohort"] = min(cohort_means)
    summary["macro_dice_maximum_cohort_gap"] = max(cohort_means) - min(cohort_means)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--baseline", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    candidate = load_cases(args.candidate)
    result = {"candidate": summarize_runs(candidate, args.bootstrap_samples), "targets": TARGETS}
    if args.baseline:
        baseline = load_cases(args.baseline)
        candidate_subjects = set.intersection(*(set(run) for run in candidate))
        baseline_subjects = set.intersection(*(set(run) for run in baseline))
        common = sorted(candidate_subjects & baseline_subjects)
        candidate_values = np.asarray(
            [np.mean([run[subject]["macro_dice"] for run in candidate]) for subject in common]
        )
        baseline_values = np.asarray(
            [np.mean([run[subject]["macro_dice"] for run in baseline]) for subject in common]
        )
        comparison = paired_bootstrap_difference(
            candidate_values, baseline_values, samples=args.bootstrap_samples
        )
        comparison["passes_plus_one_point"] = bool(
            comparison["mean_difference"] >= TARGETS["macro_dice_improvement"]
            and comparison["ci_low"] > 0
        )
        result["paired_baseline_comparison"] = comparison
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
