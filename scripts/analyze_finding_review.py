#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sample_finding_review import REVIEW_FIELDS
from tumortrust_vlm.reporting.findings import FINDING_SCHEMA_VERSION
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

LIST_FIELDS = {"lateralities", "anatomic_sites", "enhancement_patterns"}


def parse_review_value(field: str, value: str):
    value = value.strip()
    if not value or value.lower() == "unspecified":
        return None if field not in LIST_FIELDS else []
    if field in LIST_FIELDS:
        return sorted({item.strip() for item in value.split(";") if item.strip()})
    if field == "largest_reported_dimensions_mm":
        parts = [float(item.strip()) for item in value.replace("x", "*").split("*")]
        if len(parts) != 3 or any(item <= 0 for item in parts):
            raise ValueError(f"Invalid reviewer dimensions: {value!r}")
        return parts
    return value


def score_review(reviewed: list[dict], reference: dict[str, dict]) -> dict:
    scalar_correct = scalar_total = 0
    list_true_positive = list_false_positive = list_false_negative = 0
    dimension_errors = []
    by_field = {}
    for field in REVIEW_FIELDS:
        if field in LIST_FIELDS:
            true_positive = false_positive = false_negative = 0
            for row in reviewed:
                automatic = set(reference[row["case_code"]]["automatic_findings"][field])
                reviewer = set(parse_review_value(field, row[f"reviewer_{field}"]))
                true_positive += len(automatic & reviewer)
                false_positive += len(automatic - reviewer)
                false_negative += len(reviewer - automatic)
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            by_field[field] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
            list_true_positive += true_positive
            list_false_positive += false_positive
            list_false_negative += false_negative
        elif field == "largest_reported_dimensions_mm":
            paired = 0
            for row in reviewed:
                automatic = reference[row["case_code"]]["automatic_findings"][field]
                reviewer = parse_review_value(field, row[f"reviewer_{field}"])
                if automatic is not None and reviewer is not None:
                    paired += 1
                    dimension_errors.extend(
                        np.abs(np.sort(automatic) - np.sort(reviewer)).tolist()
                    )
            by_field[field] = {
                "paired_cases": paired,
                "mae_mm": float(np.mean(dimension_errors)) if dimension_errors else None,
            }
        else:
            correct = total = 0
            for row in reviewed:
                automatic = reference[row["case_code"]]["automatic_findings"][field]
                reviewer = parse_review_value(field, row[f"reviewer_{field}"])
                correct += automatic == reviewer
                total += 1
            by_field[field] = {"exact_accuracy": correct / total if total else None, "cases": total}
            scalar_correct += correct
            scalar_total += total
    list_precision = list_true_positive / (list_true_positive + list_false_positive) if list_true_positive + list_false_positive else 0.0
    list_recall = list_true_positive / (list_true_positive + list_false_negative) if list_true_positive + list_false_negative else 0.0
    return {
        "subjects": len(reviewed),
        "scalar_exact_accuracy": scalar_correct / scalar_total if scalar_total else None,
        "list_micro_precision": list_precision,
        "list_micro_recall": list_recall,
        "list_micro_f1": 2 * list_precision * list_recall / (list_precision + list_recall) if list_precision + list_recall else 0.0,
        "by_field": by_field,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score completed blinded finding-field review.")
    parser.add_argument("--review", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.review).open(encoding="utf-8", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    incomplete = [row["case_code"] for row in reviewed if row["review_complete"].lower() != "yes"]
    if incomplete:
        raise SystemExit(f"Finding review is incomplete for {len(incomplete)} cases")
    reference_rows = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    reference = {row["case_code"]: row for row in reference_rows}
    if set(reference) != {row["case_code"] for row in reviewed}:
        raise SystemExit("Review/reference case codes do not match")
    result = {
        "schema_version": FINDING_SCHEMA_VERSION,
        "reviewer_id": args.reviewer_id,
        "review_file_sha256": sha256_file(args.review),
        "reference_file_sha256": sha256_file(args.reference),
        "report_text_exported": False,
        **score_review(reviewed, reference),
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
