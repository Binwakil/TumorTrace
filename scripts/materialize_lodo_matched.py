#!/usr/bin/env python
"""Regenerate LODO-GLI/MEN/MET from the frozen `outputs/D3_zscore/resolved_config.json` base.

The LODO folds materialized by `materialize_cross_evaluations.py`'s default `configs/base.yaml`
base are stale (`normalization: robust_nonzero`, `seed: 20260810`) relative to the frozen
zscore_nonzero/seed-20260812 development baseline. This script rebuilds only the three LODO folds
(GLI/MEN/MET held out in turn, common-WT-only segmentation) from the correct base, verifies via
`changed_config_fields` that only the expected whitelisted fields differ, and validates each
resulting split manifest with `validate_split_manifest.py`'s rules before anything is queued for
training. It deliberately does NOT touch the MET source-holdout (`loso_met_*`) folds, which are a
separate, already-resolved lineage (see IMPLEMENTATION_LOG.md T5.10) with a different intended
`joint_weighting` -- regenerating them from this same base would silently change that.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.ablation import changed_config_fields
from tumortrust_vlm.config import load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_json

BASE_CONFIG_PATH = "outputs/D3_zscore/resolved_config.json"
OUTPUT_DIR = ROOT / "artifacts/private/cross_evaluations"
EXPECTED_WHITELIST = {
    "data.split_manifest",
    "data.label_mode",
    "model.out_channels",
    "evaluation.regions",
}


def fold_manifest(base_split: dict, train_ids: list[str], val_ids: list[str], test_ids: list[str], task: str) -> dict:
    cohort = {entry["subject_id"]: entry for entry in base_split["entries"]}
    entries = []
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for subject_id in ids:
            entries.append({**cohort[subject_id], "split": split, "split_reason": task})
    result = {
        "entries": entries,
        "task": task,
        "source_evaluation_split": "val",
        "contains_source_final_test": False,
        "final_test_locked": False,
    }
    result["manifest_sha256"] = sha256_json(result)
    return result


def main() -> None:
    base = load_config(BASE_CONFIG_PATH)
    assert base["project"]["seed"] == 20260812, f"expected seed 20260812, got {base['project']['seed']}"
    assert base["data"]["normalization"] == "zscore_nonzero", (
        f"expected zscore_nonzero, got {base['data']['normalization']}"
    )
    assert base["model"]["joint_weighting"] == "segmentation_only", (
        f"expected segmentation_only, got {base['model']['joint_weighting']}"
    )

    master_split = json.loads((ROOT / base["data"]["split_manifest"]).read_text(encoding="utf-8"))
    entries = master_split["entries"]

    registry = []
    for heldout in ("GLI", "MEN", "MET"):
        name = f"lodo_{heldout.lower()}_development"
        train_ids = [e["subject_id"] for e in entries if e["cohort"] != heldout and e["split"] == "train"]
        val_ids = [e["subject_id"] for e in entries if e["cohort"] != heldout and e["split"] == "val"]
        test_ids = [e["subject_id"] for e in entries if e["cohort"] == heldout and e["split"] == "val"]

        manifest = fold_manifest(master_split, train_ids, val_ids, test_ids, "common_WT_segmentation_and_burden_only_development")
        manifest_path = OUTPUT_DIR / f"{name}_split.json"
        atomic_json_dump(manifest, manifest_path)

        fold_config = json.loads(json.dumps(base))  # deep copy
        fold_config["project"]["name"] = f"TumorTrust-{name}"
        fold_config["project"]["output_dir"] = f"outputs/{name}"
        fold_config["data"]["split_manifest"] = str(manifest_path.relative_to(ROOT))
        fold_config["data"]["label_mode"] = "whole_tumor"
        fold_config["model"]["out_channels"] = 2
        fold_config["evaluation"]["regions"] = ["WT"]

        actual_changed = changed_config_fields(base, fold_config)
        unexpected = actual_changed - EXPECTED_WHITELIST
        if unexpected:
            print(
                f"CONFIG_DIFF_ASSERTION_FAILED for {name}: unexpected changed field(s) "
                f"{sorted(unexpected)}; allowed {sorted(EXPECTED_WHITELIST)}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        config_path = OUTPUT_DIR / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")

        validation = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_split_manifest.py"),
                "--manifest",
                str(manifest_path),
                "--report",
                str(OUTPUT_DIR / f"{name}_validation.json"),
            ],
            cwd=ROOT,
            check=False,
        )
        if validation.returncode != 0:
            print(f"MANIFEST_VALIDATION_FAILED for {name}", file=sys.stderr)
            raise SystemExit(1)

        registry.append(
            {
                "id": name,
                "config": str(config_path.relative_to(ROOT)),
                "task": "common_WT_segmentation_and_burden_only_development",
                "status": "ready",
                "changed_fields": sorted(actual_changed),
                "train_subjects": len(train_ids),
                "val_subjects": len(val_ids),
                "test_subjects": len(test_ids),
            }
        )
        print(f"LODO_MATCHED_OK: {name} (train={len(train_ids)} val={len(val_ids)} test={len(test_ids)})")

    atomic_json_dump(registry, OUTPUT_DIR / "lodo_matched_registry.json")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
