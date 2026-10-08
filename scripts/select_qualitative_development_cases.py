#!/usr/bin/env python
"""Freeze a metric-blind-to-appearance development qualitative case policy."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.utils import atomic_json_dump, sha256_file

INPUT = ROOT / "outputs/D3_zscore/val_cases.json"
OUTPUT = ROOT / "artifacts/private/qualitative_case_selection.json"


def tie_hash(subject_id: str) -> str:
    return hashlib.sha256(f"qualitative_policy_v1:{subject_id}".encode()).hexdigest()


def take_unique(candidates: list[dict], used: set[str]) -> dict:
    for candidate in candidates:
        if candidate["subject_id"] not in used:
            used.add(candidate["subject_id"])
            return candidate
    raise ValueError("Qualitative selection policy exhausted unique candidates")


def main() -> None:
    records = json.loads(INPUT.read_text(encoding="utf-8"))
    selected = []
    for cohort in ("GLI", "MEN", "MET"):
        cases = [record for record in records if record["cohort"] == cohort]
        median = float(np.median([record["macro_dice"] for record in cases]))
        used: set[str] = set()
        policies = {
            "typical": sorted(
                cases,
                key=lambda record: (
                    abs(record["macro_dice"] - median),
                    tie_hash(record["subject_id"]),
                ),
            ),
            "failure": sorted(
                cases,
                key=lambda record: (
                    record["macro_dice"],
                    tie_hash(record["subject_id"]),
                ),
            ),
            (
                "multifocal" if cohort == "MET" else "small_lesion"
            ): sorted(
                cases,
                key=(
                    (lambda record: (-record["target_component_count_WT"], tie_hash(record["subject_id"])))
                    if cohort == "MET"
                    else (lambda record: (record["target_volume_ml_WT"], tie_hash(record["subject_id"])))
                ),
            ),
        }
        for index, (role, candidates) in enumerate(policies.items()):
            case = take_unique(candidates, used)
            selected.append(
                {
                    "alias": f"{cohort}-{chr(ord('A') + index)}",
                    "cohort": cohort,
                    "role": role,
                    "subject_id": case["subject_id"],
                    "macro_dice": case["macro_dice"],
                    "wt_dice": case["dice_WT"],
                    "target_volume_ml_WT": case["target_volume_ml_WT"],
                    "target_component_count_WT": case["target_component_count_WT"],
                    "tie_hash": tie_hash(case["subject_id"]),
                }
            )
    payload = {
        "scope": "D3 z-score development validation only",
        "policy_version": "qualitative_policy_v1",
        "policy": {
            "typical": "closest macro Dice to the cohort median; salted hash tie-break",
            "failure": "lowest cohort macro Dice; salted hash tie-break",
            "small_lesion": "lowest ground-truth WT volume for GLI/MEN; salted hash tie-break",
            "multifocal": "highest ground-truth WT component count for MET; salted hash tie-break",
            "deduplication": "roles within a cohort must use different subjects",
        },
        "source_sha256": sha256_file(INPUT),
        "locked_test_opened": False,
        "cases": selected,
    }
    atomic_json_dump(payload, OUTPUT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
