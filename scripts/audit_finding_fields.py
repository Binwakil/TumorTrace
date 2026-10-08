#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import REPORT_DIRECTORIES
from tumortrust_vlm.data.inventory import load_report_splits
from tumortrust_vlm.reporting.findings import (
    FINDING_REVIEW_STATUS,
    FINDING_SCHEMA_VERSION,
    extract_normalized_findings,
    finding_atoms,
)
from tumortrust_vlm.utils import atomic_json_dump, sha256_json

FIELDS = (
    "multiplicity",
    "lateralities",
    "anatomic_sites",
    "enhancement_presence",
    "enhancement_patterns",
    "enhancement_degree",
    "edema",
    "midline_shift",
    "mass_effect",
    "margin",
    "largest_reported_dimensions_mm",
)


def summarize(records: list[dict]) -> dict:
    field_counts = {
        field: sum(
            record[field] is not None and record[field] != []
            for record in records
        )
        for field in FIELDS
    }
    distributions: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for field in FIELDS:
            value = record[field]
            if isinstance(value, list) and field != "largest_reported_dimensions_mm":
                distributions[field].update(value)
            elif value is not None and field != "largest_reported_dimensions_mm":
                distributions[field][str(value)] += 1
    subjects = len(records)
    return {
        "subjects": subjects,
        "field_counts": field_counts,
        "field_coverage": {
            field: count / subjects if subjects else 0.0 for field, count in field_counts.items()
        },
        "categorical_distributions": {
            field: dict(sorted(counts.items())) for field, counts in sorted(distributions.items())
        },
        "subjects_with_no_categorical_atoms": sum(not finding_atoms(record) for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit provisional normalized finding fields without exposing report text."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--output", default="artifacts/finding_field_audit.json")
    args = parser.parse_args()
    if not set(args.splits) <= {"train", "val"}:
        raise SystemExit("Only development report splits may be audited")
    config = load_config(args.config)
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligible = {entry["subject_id"] for entry in manifest["entries"]}
    report_splits = load_report_splits(config["data"]["report_split"])
    parsed = []
    identities = []
    excluded = 0
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for cohort, directory in REPORT_DIRECTORIES.items():
        path = Path(config["data"]["report_meta"]) / directory / "global_finding.json"
        reports = json.loads(path.read_text(encoding="utf-8"))
        for subject_id, text in reports.items():
            if report_splits.get(subject_id) not in args.splits:
                continue
            if subject_id not in eligible:
                excluded += 1
                continue
            fields = extract_normalized_findings(text)
            parsed.append(fields)
            by_cohort[cohort].append(fields)
            identities.append((subject_id, cohort, report_splits[subject_id]))
    result = {
        "schema_version": FINDING_SCHEMA_VERSION,
        "review_status": FINDING_REVIEW_STATUS,
        "splits": sorted(set(args.splits)),
        "eligible_identity_sha256": sha256_json(sorted(identities)),
        "excluded_by_master_manifest": excluded,
        "overall": summarize(parsed),
        "by_cohort": {
            cohort: summarize(records) for cohort, records in sorted(by_cohort.items())
        },
        "contains_report_text": False,
        "clinical_validation_complete": False,
    }
    atomic_json_dump(result, ROOT / args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
