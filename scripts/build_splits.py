#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.splits import assert_zero_overlap, build_master_split
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    inventory_path = ROOT / config["data"]["inventory"]
    records = json.loads(inventory_path.read_text(encoding="utf-8"))
    manifest = build_master_split(records, seed=int(config["project"]["seed"]))
    assert_zero_overlap(manifest)
    output = ROOT / config["data"]["split_manifest"]
    atomic_json_dump(manifest, output)
    aggregate = {
        "manifest_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"],
        "counts_by_cohort_split": manifest["counts_by_cohort_split"],
        "final_test_locked": manifest["final_test_locked"],
        "zero_overlap": True,
    }
    atomic_json_dump(aggregate, ROOT / "artifacts" / "split_audit.json")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

