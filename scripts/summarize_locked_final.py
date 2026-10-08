#!/usr/bin/env python
"""Create the prespecified, bootstrap-based locked-final summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.metrics import classification_metrics
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def load(path: Path) -> list[dict] | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def bootstrap_means(
    rows: list[dict], metrics: list[str], rng: np.random.Generator, samples: int
) -> dict[str, dict]:
    matrix = np.asarray([[row.get(metric, np.nan) for metric in metrics] for row in rows], float)
    estimates: dict[str, dict] = {}
    for column, metric in enumerate(metrics):
        values = matrix[:, column]
        valid = np.isfinite(values)
        source = values[valid]
        draws = source[rng.integers(0, len(source), size=(samples, len(source)))].mean(axis=1)
        estimates[metric] = {
            "estimate": float(source.mean()),
            "ci95": interval(draws),
            "effective_n": int(valid.sum()),
        }
    return estimates


def paired_bootstrap(
    left: list[dict], right: list[dict], metrics: list[str], rng: np.random.Generator, samples: int
) -> tuple[dict[str, dict], list[float]]:
    left_map = {row["subject_id"]: row for row in left}
    right_map = {row["subject_id"]: row for row in right}
    subjects = sorted(set(left_map) & set(right_map))
    results = {}
    p_values = []
    for metric in metrics:
        differences = np.asarray(
            [left_map[s][metric] - right_map[s][metric] for s in subjects], dtype=float
        )
        differences = differences[np.isfinite(differences)]
        draws = differences[
            rng.integers(0, len(differences), size=(samples, len(differences)))
        ].mean(axis=1)
        p_value = min(
            1.0,
            2
            * min(
                (np.count_nonzero(draws <= 0) + 1) / (samples + 1),
                (np.count_nonzero(draws >= 0) + 1) / (samples + 1),
            ),
        )
        results[metric] = {
            "mean_difference_left_minus_right": float(differences.mean()),
            "ci95": interval(draws),
            "paired_subjects": len(differences),
            "p_value": float(p_value),
        }
        p_values.append(float(p_value))
    return results, p_values


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_classification(
    rows: list[dict],
    rng: np.random.Generator,
    samples: int,
    *,
    temperature: float = 1.0,
) -> dict[str, dict]:
    probabilities = np.asarray([row["class_probability"] for row in rows], float)
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / float(temperature)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.asarray([row["class_target"] for row in rows], int)
    requested = ("balanced_accuracy", "macro_f1", "macro_auroc", "ece")
    point = classification_metrics(probabilities, labels)
    values = {metric: [] for metric in requested}
    attempts = 0
    while len(values[requested[0]]) < samples and attempts < samples * 10:
        attempts += 1
        indices = rng.integers(0, len(rows), size=len(rows))
        if len(np.unique(labels[indices])) != probabilities.shape[1]:
            continue
        result = classification_metrics(probabilities[indices], labels[indices])
        if all(np.isfinite(result[metric]) for metric in requested):
            for metric in requested:
                values[metric].append(float(result[metric]))
    if len(values[requested[0]]) != samples:
        raise RuntimeError("Could not obtain enough class-complete bootstrap replicates")
    return {
        metric: {"estimate": float(point[metric]), "ci95": interval(np.asarray(values[metric]))}
        for metric in requested
    }


def by_group(rows: list[dict], metrics: list[str], key: str) -> dict:
    output = {}
    for value in sorted({row[key] for row in rows}):
        selected = [row for row in rows if row[key] == value]
        output[value] = {
            "subjects": len(selected),
            **{
                metric: float(np.nanmean([row.get(metric, np.nan) for row in selected]))
                for metric in metrics
            },
        }
    return output


def interval_coverage(evidence_rows: list[dict], segmentation_rows: list[dict]) -> dict:
    targets = {row["subject_id"]: row for row in segmentation_rows}
    output = {}
    for region in ("WT", "TC", "ET", "SNFH"):
        region_result = {}
        for coverage, field in ((0.90, "interval_90_ml"), (0.95, "interval_95_ml")):
            covered = []
            widths = []
            for row in evidence_rows:
                interval_value = row["evidence"]["volumes"][region][field]
                if interval_value is None:
                    continue
                low, high = (float(value) for value in interval_value)
                target = float(targets[row["subject_id"]][f"target_volume_ml_{region}"])
                covered.append(low <= target <= high)
                widths.append(high - low)
            region_result[str(coverage)] = {
                "empirical_coverage": float(np.mean(covered)) if covered else None,
                "mean_width_ml": float(np.mean(widths)) if widths else None,
                "subjects": len(covered),
            }
        output[region] = region_result
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/private/locked_final")
    parser.add_argument("--protocol", default="artifacts/final_protocol_lock.json")
    parser.add_argument("--output", default="artifacts/locked_final_summary.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_final_test:
        raise SystemExit("Locked-final summarization requires explicit unlock")
    protocol_path = ROOT / args.protocol
    protocol = load(protocol_path)
    if protocol.get("status") != "frozen":
        raise SystemExit("Final protocol is not frozen")
    artifact_root = ROOT / args.root
    paths = {
        "nnunet_cases": artifact_root / "nnunet_cases.json",
        "d3_cases": artifact_root / "d3_cases.json",
        "c0_cases": artifact_root / "c0_cases.json",
        "report_cases": artifact_root / "deterministic_report_predictions.json",
        "clinical_cases": artifact_root / "deterministic_clinical_cases.json",
        "calibration": ROOT / "artifacts/development_final_evidence_calibration.json",
        "structured_evidence": ROOT
        / "artifacts/private/locked_final/structured_evidence_segmentation.json",
        "referral_summary": ROOT
        / "artifacts/private/locked_final/structured_evidence_segmentation.summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Locked-final outputs missing: {missing}")
    nnunet = load(paths["nnunet_cases"])
    d3 = load(paths["d3_cases"])
    c0 = load(paths["c0_cases"])
    reports = load(paths["report_cases"])
    clinical = load(paths["clinical_cases"])
    calibration = load(paths["calibration"])
    evidence = load(paths["structured_evidence"])
    assert isinstance(nnunet, list) and isinstance(d3, list) and isinstance(c0, list)
    assert isinstance(reports, list) and isinstance(clinical, list)
    assert isinstance(evidence, list)
    assert isinstance(calibration, dict)
    if not (len(nnunet) == len(d3) == len(c0) == len(evidence) == 320):
        raise ValueError("Structured locked-final outputs must each contain exactly 320 cases")
    if len(reports) != 141 or len(clinical) != 141:
        raise ValueError(
            "Locked-final report and clinical-metric outputs must each contain exactly 141 cases"
        )

    rng = np.random.default_rng(args.seed)
    segmentation_metrics = [
        "macro_dice",
        "dice_WT",
        "dice_TC",
        "dice_ET",
        "dice_SNFH",
        "lesion_f1",
        "small_lesion_recall",
        "laterality_correct",
        "relative_volume_error_WT",
    ]
    paired_metrics = ["macro_dice", "dice_WT", "dice_TC", "dice_ET", "dice_SNFH"]
    paired, raw_p = paired_bootstrap(nnunet, d3, paired_metrics, rng, args.bootstrap_samples)
    for metric, adjusted in zip(paired_metrics, holm(raw_p), strict=True):
        paired[metric]["holm_adjusted_p_value"] = adjusted

    report_metrics = [
        "structured_field_recall",
        "structured_field_unsupported_rate",
        "structured_field_contradiction_rate",
        "bleu_1",
        "bleu_4",
        "rouge_l",
    ]
    report_metrics = [metric for metric in report_metrics if any(metric in row for row in reports)]
    clinical_metrics = [
        metric
        for metric in (
            "radgraph_simple",
            "radgraph_partial",
            "radgraph_complete",
            "ratescore",
            "radcliq_v1",
        )
        if any(metric in row for row in clinical)
    ]
    summary = {
        "scope": "one-shot locked 320-case structured test and its frozen report subset",
        "protocol_sha256": sha256_file(protocol_path),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "unit": "patient",
            "confidence_interval": 0.95,
        },
        "segmentation": {
            "nnunet_resenc": bootstrap_means(
                nnunet, segmentation_metrics, rng, args.bootstrap_samples
            ),
            "segresnet_d3": bootstrap_means(d3, segmentation_metrics, rng, args.bootstrap_samples),
            "nnunet_minus_segresnet_paired": paired,
            "nnunet_by_cohort": by_group(nnunet, ["macro_dice", "dice_WT"], "cohort"),
            "nnunet_by_source": by_group(nnunet, ["macro_dice", "dice_WT"], "source_branch"),
        },
        "classification": {
            "source_limited_auxiliary": True,
            "calibration_temperature": float(calibration["temperature"]),
            "raw_metrics": bootstrap_classification(c0, rng, args.bootstrap_samples),
            "calibrated_metrics": bootstrap_classification(
                c0,
                rng,
                args.bootstrap_samples,
                temperature=float(calibration["temperature"]),
            ),
            "confusion_matrix": classification_metrics(
                np.asarray([row["class_probability"] for row in c0]),
                np.asarray([row["class_target"] for row in c0]),
            )["confusion_matrix"],
            "by_source": by_group(c0, ["class_correct"], "source_branch"),
        },
        "evidence_calibration": {
            "volume_interval_coverage": interval_coverage(evidence, d3),
            "referral": load(paths["referral_summary"]),
        },
        "deterministic_reporting": {
            "subjects": len(reports),
            "metrics": bootstrap_means(reports, report_metrics, rng, args.bootstrap_samples),
            "clinical_text_metrics": bootstrap_means(
                clinical, clinical_metrics, rng, args.bootstrap_samples
            ),
            "clinical_metric_warning": (
                "RadGraph, RaTE and RadCliQ are chest-derived secondary metrics; no radiologist "
                "validation or clinical utility claim is made."
            ),
        },
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    atomic_json_dump(summary, ROOT / args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
