#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.config import load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="artifacts/private/oof_folds")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads((ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8"))
    training = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in training:
        strata[(entry["cohort"], entry["source_branch"])].append(entry)
    fold_members = [[] for _ in range(args.folds)]
    rng = random.Random(config["project"]["seed"])
    for stratum in sorted(strata):
        values = list(strata[stratum])
        rng.shuffle(values)
        for index, entry in enumerate(values):
            fold_members[index % args.folds].append(entry)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    registry = []
    for fold_index in range(args.folds):
        validation_ids = {entry["subject_id"] for entry in fold_members[fold_index]}
        entries = []
        for entry in training:
            split = "val" if entry["subject_id"] in validation_ids else "train"
            entries.append({**entry, "split": split, "split_reason": f"oof_fold_{fold_index}"})
        fold_manifest = {"entries": entries, "oof_fold": fold_index, "final_test_included": False}
        fold_manifest["manifest_sha256"] = sha256_json(fold_manifest)
        manifest_path = output / f"fold_{fold_index}_split.json"
        atomic_json_dump(fold_manifest, manifest_path)
        fold_config = copy.deepcopy(config)
        fold_config["project"]["name"] = f"TumorTrust-OOF-{fold_index}"
        fold_config["project"]["output_dir"] = f"outputs/oof_fold_{fold_index}"
        fold_config["data"]["split_manifest"] = str(manifest_path)
        config_path = output / f"fold_{fold_index}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
        registry.append(
            {
                "fold": fold_index,
                "train": len(entries) - len(validation_ids),
                "held_out": len(validation_ids),
                "held_out_report_subjects": sum(entry["has_report"] for entry in fold_members[fold_index]),
                "config": str(config_path),
            }
        )
    atomic_json_dump(registry, output / "registry.json")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()

