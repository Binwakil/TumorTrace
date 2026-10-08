#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.config import load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def fold_manifest(
    base: dict,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    task: str,
    *,
    source_evaluation_split: str,
    contains_source_final_test: bool,
) -> dict:
    cohort = {entry["subject_id"]: entry for entry in base["entries"]}
    entries = []
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for subject_id in ids:
            source = cohort[subject_id]
            entries.append({**source, "split": split, "split_reason": task})
    result = {
        "entries": entries,
        "task": task,
        "source_evaluation_split": source_evaluation_split,
        "contains_source_final_test": contains_source_final_test,
        "final_test_locked": contains_source_final_test,
    }
    result["manifest_sha256"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/private/cross_evaluations")
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    base = json.loads((ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8"))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    registry = []
    if args.stage == "final" and not args.unlock_final_test:
        raise SystemExit(
            "Final cross-evaluations are locked. Freeze the protocol, then pass --unlock-final-test once."
        )
    entries = base["entries"]
    evaluation_split = "val" if args.stage == "development" else "test"
    for heldout in ("GLI", "MEN", "MET"):
        name = f"lodo_{heldout.lower()}_{args.stage}"
        train_ids = [
            entry["subject_id"]
            for entry in entries
            if entry["cohort"] != heldout and entry["split"] == "train"
        ]
        validation_ids = [
            entry["subject_id"]
            for entry in entries
            if entry["cohort"] != heldout and entry["split"] == "val"
        ]
        test_ids = [
            entry["subject_id"]
            for entry in entries
            if entry["cohort"] == heldout and entry["split"] == evaluation_split
        ]
        task = f"common_WT_segmentation_and_burden_only_{args.stage}"
        manifest = fold_manifest(
            base,
            train_ids,
            validation_ids,
            test_ids,
            task,
            source_evaluation_split=evaluation_split,
            contains_source_final_test=args.stage == "final",
        )
        manifest_path = output / f"{name}_split.json"
        atomic_json_dump(manifest, manifest_path)
        fold_config = copy.deepcopy(config)
        fold_config["project"]["name"] = f"TumorTrust-{name}"
        fold_config["project"]["output_dir"] = f"outputs/{name}"
        fold_config["data"]["split_manifest"] = str(manifest_path)
        fold_config["data"]["label_mode"] = "whole_tumor"
        fold_config["model"]["out_channels"] = 2
        fold_config["model"]["joint_weighting"] = "segmentation_only"
        fold_config["evaluation"]["regions"] = ["WT"]
        config_path = output / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
        registry.append({"id": name, "config": str(config_path), "task": task, "status": "ready"})
    met_branches = sorted({entry["source_branch"] for entry in entries if entry["cohort"] == "MET"})
    for heldout in met_branches:
        name = f"loso_met_{heldout.lower().replace('-', '_')}_{args.stage}"
        source_training = [
            entry["subject_id"]
            for entry in entries
            if entry["cohort"] == "MET"
            and entry["source_branch"] != heldout
            and entry["split"] == "train"
        ]
        test_ids = [
            entry["subject_id"]
            for entry in entries
            if entry["cohort"] == "MET"
            and entry["source_branch"] == heldout
            and entry["split"] == evaluation_split
        ]
        if not source_training or not test_ids:
            registry.append(
                {
                    "id": name,
                    "status": "infeasible",
                    "reason": (
                        "No leakage-safe source-training cases or held-out evaluation cases exist "
                        f"for stage={args.stage}. Source branch and the locked report split are confounded."
                    ),
                }
            )
            continue
        validation_ids = [
            subject_id
            for subject_id in source_training
            if int(hashlib.sha256(subject_id.encode()).hexdigest(), 16) % 10 == 0
        ]
        validation_set = set(validation_ids)
        train_ids = [
            subject_id for subject_id in source_training if subject_id not in validation_set
        ]
        manifest = fold_manifest(
            base,
            train_ids,
            validation_ids,
            test_ids,
            "MET_source_holdout",
            source_evaluation_split=evaluation_split,
            contains_source_final_test=args.stage == "final",
        )
        manifest_path = output / f"{name}_split.json"
        atomic_json_dump(manifest, manifest_path)
        fold_config = copy.deepcopy(config)
        fold_config["project"]["name"] = f"TumorTrust-{name}"
        fold_config["project"]["output_dir"] = f"outputs/{name}"
        fold_config["data"]["split_manifest"] = str(manifest_path)
        config_path = output / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
        registry.append(
            {
                "id": name,
                "config": str(config_path),
                "task": "MET_source_holdout",
                "status": "ready",
            }
        )
    atomic_json_dump(registry, output / "registry.json")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
