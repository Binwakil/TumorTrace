#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.inventory import build_inventory
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def render_audit(audit: dict, inventory_hash: str) -> str:
    lines = [
        "# TumorTrust-VLM Data Inventory Audit",
        "",
        f"Inventory SHA-256: `{inventory_hash}`",
        "",
        "## Counts",
        "",
        f"- Labeled, four-sequence complete: **{audit['labeled_complete']}**",
        f"- Unlabeled, four-sequence complete: **{audit['unlabeled_complete']}**",
        f"- Labeled by cohort: `{audit['labeled_by_cohort']}`",
        f"- Unlabeled by cohort: `{audit['unlabeled_by_cohort']}`",
        f"- Corrected MEN copies selected: **{audit['corrected_copies']}**",
        f"- Report-matched subjects: **{audit['report_subjects']}**",
        f"- Report splits: `{audit['report_splits']}`",
        f"- Labeled orientations: `{audit['orientation_labeled']}`",
        "",
        "## Integrity gates",
        "",
        f"- Incomplete image sets: **{len(audit['incomplete_images'])}**",
        f"- Branch/label mismatches: **{len(audit['unexpected_label_branch_mismatch'])}**",
        f"- Alignment failures: **{len(audit['alignment_failures'])}**",
        f"- Cross-case content-fingerprint groups: **{len(audit['content_duplicate_groups'])}**",
        f"- Duplicate resolution: `{audit['duplicate_resolution_counts']}`",
        f"- Labeled cases eligible after resolution: **{audit['eligible_labeled_after_duplicate_resolution']}**",
        f"- Observed label sets: `{audit['label_value_sets']}`",
        "",
        "Subject-level paths, IDs, and fingerprints are stored only in the ignored private artifact.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fingerprints", action="store_true")
    parser.add_argument("--deep-label-audit", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    records, audit = build_inventory(
        data["dataset_root"],
        data["report_meta"],
        data["report_split"],
        fingerprints=args.fingerprints,
        deep_label_audit=args.deep_label_audit,
    )
    output = ROOT / data["inventory"]
    atomic_json_dump(records, output)
    inventory_hash = sha256_json(records)
    audit["inventory_sha256"] = inventory_hash
    atomic_json_dump(audit, ROOT / "artifacts" / "private" / "inventory_audit_detailed.json")
    public_audit = {
        key: value
        for key, value in audit.items()
        if key
        not in {
            "missing_report_cases",
            "incomplete_images",
            "unexpected_label_branch_mismatch",
            "alignment_failures",
            "content_duplicate_groups",
            "duplicate_resolutions",
        }
    }
    public_audit["content_duplicate_group_count"] = len(audit["content_duplicate_groups"])
    public_audit["alignment_failure_count"] = len(audit["alignment_failures"])
    atomic_json_dump(public_audit, ROOT / "artifacts" / "inventory_audit.json")
    audit_path = ROOT / "artifacts" / "inventory_audit.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(render_audit(audit, inventory_hash), encoding="utf-8")
    print(render_audit(audit, inventory_hash))


if __name__ == "__main__":
    main()
