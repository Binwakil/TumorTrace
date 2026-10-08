#!/usr/bin/env python
"""Validate a development cross-evaluation split manifest before it is used for training.

Enforces the non-negotiable rules for any manifest that uses a "test" role during development:
  - `contains_source_final_test` must be explicitly `false`.
  - Zero subject-ID overlap with the master split's locked final-test subjects.
  - No subject appears in more than one role within the candidate manifest itself.
  - Reports per-role, per-cohort counts for the run log.

Exits nonzero with an explicit message on any violation; prints a PASSED summary and writes a JSON
report on success. This does not open, read, or otherwise touch the master final-test subjects'
data -- it only compares subject_id sets already present in the (already-materialized) manifests.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Candidate development split manifest to validate.")
    parser.add_argument(
        "--master-split",
        default="artifacts/private/master_split.json",
        help="Master split manifest whose 'test' role is the locked 320-subject final test.",
    )
    parser.add_argument("--report", help="Optional path to write a JSON validation report.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    master = json.loads(Path(args.master_split).read_text(encoding="utf-8"))

    failures: list[str] = []

    contains_final_test = manifest.get("contains_source_final_test")
    if contains_final_test is not False:
        failures.append(
            f"contains_source_final_test must be explicit false, got {contains_final_test!r}"
        )

    master_final_test_ids = {
        entry["subject_id"] for entry in master["entries"] if entry["split"] == "test"
    }
    if len(master_final_test_ids) != 320:
        failures.append(
            f"master split's locked final test has {len(master_final_test_ids)} subjects, "
            "expected 320 -- master_split.json may not be the expected file; refusing to proceed."
        )

    role_ids: dict[str, set[str]] = {}
    role_cohort_counts: dict[str, Counter] = {}
    seen_anywhere: dict[str, str] = {}
    cross_role_duplicates: list[str] = []
    for entry in manifest["entries"]:
        subject_id = entry["subject_id"]
        split = entry["split"]
        role_ids.setdefault(split, set()).add(subject_id)
        role_cohort_counts.setdefault(split, Counter())[entry["cohort"]] += 1
        if subject_id in seen_anywhere and seen_anywhere[subject_id] != split:
            cross_role_duplicates.append(f"{subject_id} in both {seen_anywhere[subject_id]!r} and {split!r}")
        seen_anywhere[subject_id] = split

    if cross_role_duplicates:
        failures.append(
            f"{len(cross_role_duplicates)} subject(s) appear in more than one split role: "
            f"{cross_role_duplicates[:10]}{'...' if len(cross_role_duplicates) > 10 else ''}"
        )

    overlap_by_role = {}
    for role, ids in role_ids.items():
        overlap = ids & master_final_test_ids
        overlap_by_role[role] = sorted(overlap)
        if overlap:
            failures.append(
                f"{len(overlap)} subject(s) in manifest role {role!r} overlap the locked master "
                f"final test: {sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
            )

    report = {
        "manifest": args.manifest,
        "master_split": args.master_split,
        "contains_source_final_test": contains_final_test,
        "master_final_test_subjects": len(master_final_test_ids),
        "role_counts": {role: dict(counts) for role, counts in role_cohort_counts.items()},
        "role_totals": {role: len(ids) for role, ids in role_ids.items()},
        "cross_role_duplicate_count": len(cross_role_duplicates),
        "overlap_with_master_final_test": overlap_by_role,
        "passed": not failures,
        "failures": failures,
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        print("MANIFEST_VALIDATION_FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("MANIFEST_VALIDATION_PASSED")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
