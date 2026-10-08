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
from tumortrust_vlm.utils import atomic_json_dump


def majority_lookup_accuracy(records: list[dict], field: str) -> float:
    groups: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        groups[str(record[field])][record["cohort"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(records)


def contingency(records: list[dict], field: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        groups[str(record[field])][record["cohort"]] += 1
    return {name: dict(counts) for name, counts in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    inventory = json.loads(
        (ROOT / config["data"]["inventory"]).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8")
    )
    by_id = {record["subject_id"]: record for record in inventory}
    records = [
        {
            **by_id[entry["subject_id"]],
            "split": entry["split"],
        }
        for entry in manifest["entries"]
    ]
    result = {
        "interpretation": (
            "These are shortcut-risk diagnostics, not evidence of biological classification. "
            "A high majority-lookup accuracy means acquisition metadata can predict cohort."
        ),
        "all": {
            "source_majority_lookup_accuracy": majority_lookup_accuracy(
                records, "source_branch"
            ),
            "orientation_majority_lookup_accuracy": majority_lookup_accuracy(
                records, "orientation"
            ),
            "source_by_cohort": contingency(records, "source_branch"),
            "orientation_by_cohort": contingency(records, "orientation"),
        },
        "by_split": {},
    }
    for split in ("train", "val", "test"):
        selected = [record for record in records if record["split"] == split]
        result["by_split"][split] = {
            "subjects": len(selected),
            "source_majority_lookup_accuracy": majority_lookup_accuracy(
                selected, "source_branch"
            ),
            "orientation_majority_lookup_accuracy": majority_lookup_accuracy(
                selected, "orientation"
            ),
            "source_by_cohort": contingency(selected, "source_branch"),
            "orientation_by_cohort": contingency(selected, "orientation"),
        }
    atomic_json_dump(result, ROOT / "artifacts" / "shortcut_audit.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
