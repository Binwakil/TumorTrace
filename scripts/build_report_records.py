#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import REPORT_DIRECTORIES
from tumortrust_vlm.data.inventory import load_report_splits
from tumortrust_vlm.reporting.targets import assemble_report_records
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_final_test:
        raise SystemExit("Final test report records are locked")
    config = load_config(args.config)
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligible_subject_ids = {entry["subject_id"] for entry in manifest["entries"]}
    feature_map = {}
    duplicates = []
    for path in args.features:
        for record in json.loads(Path(path).read_text(encoding="utf-8")):
            if record["subject_id"] in feature_map:
                duplicates.append(record["subject_id"])
            feature_map[record["subject_id"]] = record
    if duplicates:
        raise SystemExit(f"Subjects have more than one prediction source: {duplicates[:5]}")
    split_map = load_report_splits(config["data"]["report_split"])
    reports = {}
    for directory in REPORT_DIRECTORIES.values():
        path = Path(config["data"]["report_meta"]) / directory / "global_finding.json"
        reports.update(json.loads(path.read_text(encoding="utf-8")))
    try:
        records, excluded = assemble_report_records(
            feature_map,
            reports,
            split_map,
            eligible_subject_ids,
            args.split,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    atomic_json_dump(records, args.output)
    print(
        json.dumps(
            {
                "split": args.split,
                "subjects": len(records),
                "excluded_by_master_manifest": len(excluded),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
