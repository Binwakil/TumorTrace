#!/usr/bin/env python
"""T9.4/T9.5/T9.6: K-fold out-of-fold (OOF) evidence folds for reporter training.

Every one of the report-training subjects (master_split.json's has_report==True, split=="train")
needs evidence (segmentation, volumetry, classification) from a core model that did NOT train on
that subject -- otherwise the reporter could learn to trust suspiciously-perfect evidence for
subjects the core model memorized, rather than genuinely evidence-grounded reporting.

User decision (2026-08-16): evidence is COMPOSED from two single-task recipes rather than one
multitask model, consistent with M4's own finding that joint training measurably costs segmentation
quality (IMPLEMENTATION_LOG.md, T5.6). Segmentation/volumetry evidence comes from `segmentation_only`
OOF folds matched to `outputs/D3_zscore/resolved_config.json` (the frozen best single-task
segmentation baseline); classification-probability evidence comes from separate `classification_only`
OOF folds matched to `outputs/C0/resolved_config.json`. Each is single-task, so each individual fold
trains faster than a multitask run would, even though there are two fold sets instead of one.

For each of K folds, for each recipe: train on {all 1915 master-train subjects} MINUS {this fold's
held-out report-training subjects}, matched field-for-field to that recipe's frozen baseline (only
data.split_manifest differs, whitelist-asserted). Report-validation (69) and report-test (141)
subjects are never in any fold's training set -- already locked out by the master split (T2.6/T2.7);
this script does not touch that lock. A companion "report-val" fold per recipe (full report-train set
kept in training) generates frozen evidence for the 69 report-validation subjects (T9.6).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.ablation import changed_config_fields
from tumortrust_vlm.config import load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_json

K = 5
RECIPES = {
    "seg": "outputs/D3_zscore/resolved_config.json",  # segmentation_only: segmentation/volumetry evidence
    "cls": "outputs/C0/resolved_config.json",  # classification_only: class-probability evidence
}
OUTPUT_DIR = ROOT / "artifacts/private/oof_folds"


def fold_manifest(master: dict, train_ids: list[str], val_ids: list[str], test_ids: list[str], task: str) -> dict:
    by_id = {e["subject_id"]: e for e in master["entries"]}
    entries = []
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for subject_id in ids:
            entries.append({**by_id[subject_id], "split": split, "split_reason": task})
    result = {
        "entries": entries,
        "task": task,
        "source_evaluation_split": "test",
        "contains_source_final_test": False,
        "final_test_locked": False,
    }
    result["manifest_sha256"] = sha256_json(result)
    return result


def build_recipe(recipe_name: str, base_config_path: str, report_train_ids: list[str],
                  other_train_ids: list[str], non_report_val_ids: list[str],
                  report_val_ids: list[str], master: dict) -> list[dict]:
    base = load_config(base_config_path)
    assert base["project"]["seed"] == 20260812, f"{recipe_name}: expected seed 20260812"
    assert base["data"]["normalization"] == "zscore_nonzero", f"{recipe_name}: expected zscore_nonzero"

    fold_of = {
        subject_id: int(hashlib.sha256(f"{recipe_name}:{subject_id}".encode()).hexdigest(), 16) % K
        for subject_id in report_train_ids
    }
    registry = []
    all_held_out = []

    for fold in range(K):
        name = f"oof_{recipe_name}_fold_{fold}_development"
        held_out = [s for s in report_train_ids if fold_of[s] == fold]
        kept_report = [s for s in report_train_ids if fold_of[s] != fold]
        train_ids = other_train_ids + kept_report
        manifest = fold_manifest(master, train_ids, non_report_val_ids, held_out, f"oof_evidence_fold_{recipe_name}")
        manifest_path = OUTPUT_DIR / f"{name}_split.json"
        atomic_json_dump(manifest, manifest_path)
        all_held_out.extend(held_out)

        fold_config = json.loads(json.dumps(base))
        fold_config["project"]["name"] = f"TumorTrust-{name}"
        fold_config["project"]["output_dir"] = f"outputs/{name}"
        fold_config["data"]["split_manifest"] = str(manifest_path.relative_to(ROOT))
        actual_changed = changed_config_fields(base, fold_config)
        if actual_changed != {"data.split_manifest"}:
            print(f"CONFIG_DIFF_ASSERTION_FAILED for {name}: {sorted(actual_changed)}", file=sys.stderr)
            raise SystemExit(1)
        config_path = OUTPUT_DIR / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")

        registry.append({
            "id": name, "recipe": recipe_name, "fold": fold,
            "config": str(config_path.relative_to(ROOT)),
            "held_out_report_subjects": len(held_out),
            "train_subjects": len(train_ids),
        })
        print(f"OOF_FOLD_OK: {name} (held_out={len(held_out)} train={len(train_ids)})")

    missing = set(report_train_ids) - set(all_held_out)
    duplicated = len(all_held_out) - len(set(all_held_out))
    if missing or duplicated:
        print(
            f"OOF_COVERAGE_FAILED [{recipe_name}]: missing={len(missing)} duplicated={duplicated}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"OOF_COVERAGE_OK [{recipe_name}]: all {len(report_train_ids)} report-train subjects "
          f"held out exactly once across {K} folds")

    # Companion report-val fold: full report-train set kept in training (69 report-val subjects
    # never trained on regardless), used to generate T9.6's frozen validation-panel evidence.
    name = f"oof_{recipe_name}_reportval_development"
    train_ids = other_train_ids + report_train_ids
    manifest = fold_manifest(master, train_ids, non_report_val_ids, report_val_ids, f"oof_evidence_reportval_{recipe_name}")
    manifest_path = OUTPUT_DIR / f"{name}_split.json"
    atomic_json_dump(manifest, manifest_path)
    fold_config = json.loads(json.dumps(base))
    fold_config["project"]["name"] = f"TumorTrust-{name}"
    fold_config["project"]["output_dir"] = f"outputs/{name}"
    fold_config["data"]["split_manifest"] = str(manifest_path.relative_to(ROOT))
    actual_changed = changed_config_fields(base, fold_config)
    if actual_changed != {"data.split_manifest"}:
        print(f"CONFIG_DIFF_ASSERTION_FAILED for {name}: {sorted(actual_changed)}", file=sys.stderr)
        raise SystemExit(1)
    config_path = OUTPUT_DIR / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
    registry.append({
        "id": name, "recipe": recipe_name, "fold": None,
        "config": str(config_path.relative_to(ROOT)),
        "held_out_report_subjects": len(report_val_ids),
        "train_subjects": len(train_ids),
    })
    print(f"OOF_REPORTVAL_OK [{recipe_name}]: {name} (held_out={len(report_val_ids)} train={len(train_ids)})")
    return registry


def main() -> None:
    reference = load_config(RECIPES["seg"])
    master = json.loads((ROOT / reference["data"]["split_manifest"]).read_text(encoding="utf-8"))
    entries = master["entries"]

    report_train_ids = sorted(e["subject_id"] for e in entries if e.get("has_report") and e["split"] == "train")
    report_train_set = set(report_train_ids)
    other_train_ids = [e["subject_id"] for e in entries if e["split"] == "train" and e["subject_id"] not in report_train_set]
    non_report_val_ids = [e["subject_id"] for e in entries if e["split"] == "val" and not e.get("has_report")]
    report_val_ids = sorted(e["subject_id"] for e in entries if e.get("has_report") and e["split"] == "val")

    print(f"report_train={len(report_train_ids)} other_train={len(other_train_ids)} "
          f"report_val={len(report_val_ids)}")

    full_registry = []
    for recipe_name, base_config_path in RECIPES.items():
        full_registry.extend(
            build_recipe(recipe_name, base_config_path, report_train_ids, other_train_ids,
                         non_report_val_ids, report_val_ids, master)
        )
    atomic_json_dump(full_registry, OUTPUT_DIR / "oof_registry.json")
    print(json.dumps(full_registry, indent=2))


if __name__ == "__main__":
    main()
