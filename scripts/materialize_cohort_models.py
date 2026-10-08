#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import COHORTS
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def cohort_manifest(master: dict, cohort: str) -> dict:
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort: {cohort}")
    entries = [
        {**entry, "split_reason": f"cohort_specific_{cohort}_development"}
        for entry in master["entries"]
        if entry["cohort"] == cohort and entry["split"] in {"train", "val"}
    ]
    result = {
        "entries": entries,
        "task": "cohort_specific_segmentation_development",
        "cohort": cohort,
        "source_evaluation_split": "val",
        "contains_source_final_test": False,
        "final_test_locked": False,
    }
    result["manifest_sha256"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize development-only cohort-specific segmentation controls."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/private/cohort_specific")
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    master = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    registry = []
    for cohort in COHORTS:
        name = f"cohort_specific_{cohort.lower()}"
        manifest = cohort_manifest(master, cohort)
        cohort_manifest_path = output / f"{name}_split.json"
        atomic_json_dump(manifest, cohort_manifest_path)
        fold_config = copy.deepcopy(config)
        fold_config["project"]["name"] = f"TumorTrust-{name}"
        fold_config["project"]["output_dir"] = f"outputs/{name}"
        fold_config["data"]["split_manifest"] = str(cohort_manifest_path)
        fold_config["data"]["class_balanced_sampler"] = False
        fold_config["data"]["missing_modality_probability"] = 0.0
        fold_config["model"]["joint_weighting"] = "segmentation_only"
        fold_config["training"]["max_steps_per_epoch"] = args.steps_per_epoch
        fold_config["training"]["validate_every"] = args.validate_every
        config_path = output / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
        counts = {
            split: sum(entry["split"] == split for entry in manifest["entries"])
            for split in ("train", "val")
        }
        registry.append(
            {
                "id": name,
                "cohort": cohort,
                "config": str(config_path),
                **counts,
                "status": "ready",
            }
        )
    atomic_json_dump(registry, output / "registry.json")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
