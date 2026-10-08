from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from tumortrust_vlm.evaluation.statistics import paired_bootstrap_difference

BINARY_FIELDS = (
    "clinically_significant_error_or_omission",
    "potentially_harmful_error",
    "unsupported_statement",
    "quantitative_contradiction",
    "laterality_contradiction",
)

AGREEMENT_FIELDS = (
    "factual_correctness",
    "completeness",
    *BINARY_FIELDS,
    "required_editing",
)


def fleiss_kappa(item_ratings: list[list[object]]) -> float:
    """Compute Fleiss' kappa for items rated by the same number of readers."""
    if not item_ratings or min(map(len, item_ratings)) < 2:
        raise ValueError("Fleiss kappa requires at least one item and two readers")
    reader_counts = {len(ratings) for ratings in item_ratings}
    if len(reader_counts) != 1:
        raise ValueError("Fleiss kappa requires equal reader counts per item")
    readers = reader_counts.pop()
    categories = sorted({rating for ratings in item_ratings for rating in ratings}, key=str)
    matrix = np.asarray(
        [[Counter(ratings)[category] for category in categories] for ratings in item_ratings],
        dtype=float,
    )
    item_agreement = (np.square(matrix).sum(1) - readers) / (readers * (readers - 1))
    category_frequency = matrix.sum(0) / matrix.sum()
    expected = float(np.square(category_frequency).sum())
    observed = float(item_agreement.mean())
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _validated_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a JSON boolean")
    return value


def summarize_radiologist_scores(
    scores: list[dict], private_map: list[dict], bootstrap_samples: int = 2000
) -> dict:
    mapping = {row["blinded_report_id"]: row for row in private_map}
    joined = []
    for score in scores:
        blinded_id = score["blinded_report_id"]
        if blinded_id not in mapping:
            raise ValueError(f"Unknown blinded report ID: {blinded_id}")
        joined.append({**score, **mapping[blinded_id]})
    if not joined:
        raise ValueError("No radiologist scores were provided")

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        by_method[row["method"]].append(row)
    method_summary = {}
    for method, rows in sorted(by_method.items()):
        significant = np.asarray(
            [
                _validated_boolean(
                    row["clinically_significant_error_or_omission"],
                    "clinically_significant_error_or_omission",
                )
                for row in rows
            ],
            dtype=float,
        )
        harmful = np.asarray(
            [
                _validated_boolean(row["potentially_harmful_error"], "potentially_harmful_error")
                for row in rows
            ],
            dtype=float,
        )
        method_summary[method] = {
            "ratings": len(rows),
            "subjects": len({row["subject_id"] for row in rows}),
            "no_clinically_significant_error_fraction": float(1 - significant.mean()),
            "harmful_error_rate": float(harmful.mean()),
            "mean_usefulness": float(np.mean([float(row["usefulness_1_to_5"]) for row in rows])),
            "passes_no_significant_error_gate": bool(1 - significant.mean() >= 0.80),
            "passes_harmful_error_gate": bool(harmful.mean() <= 0.05),
        }

    agreement = {}
    by_report: dict[str, list[dict]] = defaultdict(list)
    for row in joined:
        by_report[row["blinded_report_id"]].append(row)
    for field in AGREEMENT_FIELDS:
        ratings = [
            [row[field] for row in rows]
            for rows in by_report.values()
            if len(rows) >= 2 and all(row.get(field) is not None for row in rows)
        ]
        if not ratings:
            agreement[field] = {"kappa": None, "status": "insufficient_repeat_ratings"}
        elif len({len(values) for values in ratings}) != 1:
            agreement[field] = {"kappa": None, "status": "unequal_reader_counts"}
        else:
            agreement[field] = {
                "kappa": float(fleiss_kappa(ratings)),
                "items": len(ratings),
                "readers_per_item": len(ratings[0]),
                "status": "complete",
            }

    subject_method_significant: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in joined:
        subject_method_significant[(row["subject_id"], row["method"])].append(
            float(
                _validated_boolean(
                    row["clinically_significant_error_or_omission"],
                    "clinically_significant_error_or_omission",
                )
            )
        )
    paired = {}
    methods = sorted(by_method)
    for left, right in combinations(methods, 2):
        subjects = sorted(
            {subject for subject, method in subject_method_significant if method == left}
            & {subject for subject, method in subject_method_significant if method == right}
        )
        if not subjects:
            continue
        left_values = np.asarray(
            [np.mean(subject_method_significant[(subject, left)]) for subject in subjects]
        )
        right_values = np.asarray(
            [np.mean(subject_method_significant[(subject, right)]) for subject in subjects]
        )
        paired[f"{left}_minus_{right}"] = {
            "subjects": len(subjects),
            "metric": "clinically_significant_error_rate",
            **paired_bootstrap_difference(left_values, right_values, samples=bootstrap_samples),
        }
    return {
        "ratings": len(joined),
        "methods": method_summary,
        "agreement": agreement,
        "paired_method_comparisons": paired,
    }
