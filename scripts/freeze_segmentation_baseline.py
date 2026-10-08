#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.statistics import paired_bootstrap_difference
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

PAIRED_METRICS = (
    "macro_dice",
    "dice_WT",
    "dice_TC",
    "dice_ET",
    "dice_SNFH",
    "hd95_WT",
    "hd95_TC",
    "hd95_ET",
    "hd95_SNFH",
    "surface_dice_WT",
    "surface_dice_TC",
    "surface_dice_ET",
    "surface_dice_SNFH",
    "lesion_f1",
    "small_lesion_recall",
    "relative_volume_error_WT",
    "component_count_absolute_error_WT",
    "laterality_correct",
)
LOWER_IS_BETTER = {
    "hd95_WT",
    "hd95_TC",
    "hd95_ET",
    "hd95_SNFH",
    "relative_volume_error_WT",
    "component_count_absolute_error_WT",
}


def load_cases(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    by_subject = {record["subject_id"]: record for record in records}
    if len(by_subject) != len(records):
        raise ValueError(f"Duplicate subject IDs in {path}")
    return by_subject


def validation_subjects(path: Path) -> set[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {entry["subject_id"] for entry in manifest["entries"] if entry["split"] == "val"}


def metric_values(cases: dict[str, dict], subjects: list[str], metric: str) -> np.ndarray:
    missing = [subject for subject in subjects if metric not in cases[subject]]
    if missing:
        raise ValueError(f"Metric {metric} is missing for {len(missing)} subjects")
    return np.asarray([cases[subject][metric] for subject in subjects], dtype=float)


def finite_mean(values: np.ndarray, metric: str) -> tuple[float, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"Metric {metric} has no finite observations")
    return float(finite.mean()), len(finite)


def summarize_candidate(cases: dict[str, dict], subjects: list[str]) -> dict:
    summary = {}
    effective_n = {}
    for metric in PAIRED_METRICS:
        summary[metric], effective_n[metric] = finite_mean(
            metric_values(cases, subjects, metric), metric
        )
    summary["effective_n"] = effective_n
    cohorts = sorted({cases[subject]["cohort"] for subject in subjects})
    summary["macro_dice_by_cohort"] = {
        cohort: float(
            np.mean(
                [
                    cases[subject]["macro_dice"]
                    for subject in subjects
                    if cases[subject]["cohort"] == cohort
                ]
            )
        )
        for cohort in cohorts
    }
    return summary


def paired_comparison(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    subjects: list[str],
    bootstrap_samples: int,
) -> dict:
    result = {}
    for metric in PAIRED_METRICS:
        candidate_values = metric_values(candidate, subjects, metric)
        baseline_values = metric_values(baseline, subjects, metric)
        paired_finite = np.isfinite(candidate_values) & np.isfinite(baseline_values)
        paired_n = int(paired_finite.sum())
        if not paired_n:
            raise ValueError(f"Metric {metric} has no paired finite observations")
        comparison = paired_bootstrap_difference(
            candidate_values[paired_finite],
            baseline_values[paired_finite],
            samples=bootstrap_samples,
        )
        lower = metric in LOWER_IS_BETTER
        result[metric] = {
            **comparison,
            "paired_n": paired_n,
            "direction": "lower_is_better" if lower else "higher_is_better",
            "mean_improvement": (
                -comparison["mean_difference"] if lower else comparison["mean_difference"]
            ),
            "improvement_ci_low": (
                -comparison["ci_high"] if lower else comparison["ci_low"]
            ),
            "improvement_ci_high": (
                -comparison["ci_low"] if lower else comparison["ci_high"]
            ),
        }
    return result


def freeze_selection(
    candidates: list[tuple[str, Path, Path, Path]],
    split_manifest: Path,
    bootstrap_samples: int,
    hd95_tolerance_mm: float,
) -> dict:
    if len(candidates) < 2:
        raise ValueError("At least two segmentation candidates are required")
    names = [candidate[0] for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique")
    expected = validation_subjects(split_manifest)
    loaded = {name: load_cases(cases) for name, cases, _, _ in candidates}
    for name, cases in loaded.items():
        if set(cases) != expected:
            missing = len(expected - set(cases))
            unexpected = len(set(cases) - expected)
            raise ValueError(
                f"{name} does not exactly match frozen validation subjects: "
                f"missing={missing}, unexpected={unexpected}"
            )
    subjects = sorted(expected)
    summaries = {name: summarize_candidate(loaded[name], subjects) for name in names}
    ranking = sorted(names, key=lambda name: summaries[name]["macro_dice"], reverse=True)
    selected, runner_up = ranking[:2]
    comparisons = {
        f"{selected}_vs_{other}": paired_comparison(
            loaded[selected], loaded[other], subjects, bootstrap_samples
        )
        for other in ranking[1:]
    }
    primary = comparisons[f"{selected}_vs_{runner_up}"]["macro_dice"]
    hd95 = comparisons[f"{selected}_vs_{runner_up}"]["hd95_WT"]
    cohort_regressions = {
        cohort: summaries[selected]["macro_dice_by_cohort"][cohort]
        - summaries[runner_up]["macro_dice_by_cohort"][cohort]
        for cohort in summaries[selected]["macro_dice_by_cohort"]
    }
    provenance = {
        name: {
            "cases": str(cases),
            "cases_sha256": sha256_file(cases),
            "config": str(config),
            "config_sha256": sha256_file(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        for name, cases, config, checkpoint in candidates
    }
    warnings = []
    if hd95["mean_difference"] > hd95_tolerance_mm:
        warnings.append(
            f"Selected candidate WT HD95 is {hd95['mean_difference']:.3f} mm worse than runner-up"
        )
    regressed = {cohort: value for cohort, value in cohort_regressions.items() if value < -0.01}
    if regressed:
        warnings.append(f"Selected candidate regresses cohort macro Dice by >1 point: {regressed}")
    return {
        "schema_version": 1,
        "selection_scope": "frozen_development_validation_only",
        "locked_test_opened": False,
        "paired_subjects": len(subjects),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "primary_metric": "macro_dice",
        "ranking": ranking,
        "selected_candidate": selected,
        "runner_up": runner_up,
        "observed_best_selected": True,
        "primary_superiority_supported": bool(primary["improvement_ci_low"] > 0),
        "wt_hd95_within_tolerance": bool(hd95["mean_difference"] <= hd95_tolerance_mm),
        "hd95_tolerance_mm": hd95_tolerance_mm,
        "cohort_macro_dice_differences_vs_runner_up": cohort_regressions,
        "warnings": warnings,
        "candidates": summaries,
        "paired_comparisons": comparisons,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the best same-subject development segmentation baseline."
    )
    parser.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        metavar=("NAME", "CASES", "CONFIG", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument(
        "--split-manifest", default="artifacts/private/master_split.json"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--hd95-tolerance-mm", type=float, default=0.5)
    parser.add_argument(
        "--output", default="artifacts/development_segmentation_baseline_selection.json"
    )
    args = parser.parse_args()
    candidates = [
        (name, Path(cases), Path(config), Path(checkpoint))
        for name, cases, config, checkpoint in args.candidate
    ]
    result = freeze_selection(
        candidates,
        Path(args.split_manifest),
        args.bootstrap_samples,
        args.hd95_tolerance_mm,
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
