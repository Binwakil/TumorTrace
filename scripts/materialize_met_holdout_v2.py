#!/usr/bin/env python
"""Redesigned MET source-branch holdout (T5.10 follow-up).

The original `loso_met_*` fold (`materialize_cross_evaluations.py`) trains and evaluates on
MET-cohort subjects only -- both its "val" and "test" roles contain zero GLI/MEN subjects, so a
classifier trained on that fold never sees GLI/MEN labels and its balanced-accuracy numbers cannot
show cross-cohort confusion under source shift by construction. This script builds a fold that keeps
the held-out-MET-source-branch design but trains and evaluates alongside the standard GLI/MEN split,
so the >=0.75 balanced-accuracy gate in PLAN.md Sec17.3 is actually tested: does the classifier still
call held-out-source-branch MET subjects "MET" rather than confusing them with GLI/MEN, given it saw
GLI/MEN normally during training?

- train: MET (non-held-out source branch, master train split) + all GLI (master train split) +
  all MEN (master train split) -- i.e. the standard training recipe, minus one MET source branch.
- val (training-time early-stopping panel): a small hash-subsample of the same pool, mixed across
  cohorts so early stopping is not MET-only.
- test (the evaluation role read by `evaluate.py --split test`; per `split_requires_final_unlock`,
  this manifest sets `contains_source_final_test: False` so it needs no `--unlock-final-test`, exactly
  matching every other development LODO/LOSO manifest's already-documented convention): held-out-
  source-branch MET subjects (master val split) UNION the standard GLI/MEN master val split -- a
  genuine 3-class panel that can show source-shift-driven cross-cohort confusion if it exists.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.config import load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def main() -> None:
    config = load_config("configs/base.yaml")
    base = json.loads(
        (ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8")
    )
    entries = base["entries"]
    by_id = {entry["subject_id"]: entry for entry in entries}
    output = ROOT / "artifacts/private/cross_evaluations"
    output.mkdir(parents=True, exist_ok=True)

    met_branches = sorted({e["source_branch"] for e in entries if e["cohort"] == "MET"})
    registry = []
    for heldout in met_branches:
        name = f"loso_met_{heldout.lower().replace('-', '_')}_v2_development"
        met_train_pool = [
            e["subject_id"]
            for e in entries
            if e["cohort"] == "MET" and e["source_branch"] != heldout and e["split"] == "train"
        ]
        met_test = [
            e["subject_id"]
            for e in entries
            if e["cohort"] == "MET" and e["source_branch"] == heldout and e["split"] == "val"
        ]
        if not met_train_pool or not met_test:
            registry.append({"id": name, "status": "infeasible", "reason": "no leakage-safe cases"})
            continue
        other_train = [
            e["subject_id"] for e in entries if e["cohort"] in ("GLI", "MEN") and e["split"] == "train"
        ]
        other_val = [
            e["subject_id"] for e in entries if e["cohort"] in ("GLI", "MEN") and e["split"] == "val"
        ]

        def held_back(pool: list[str], fraction_mod: int) -> tuple[list[str], list[str]]:
            held = [
                s for s in pool if int(hashlib.sha256(s.encode()).hexdigest(), 16) % fraction_mod == 0
            ]
            held_set = set(held)
            return [s for s in pool if s not in held_set], held

        met_train, met_earlystop = held_back(met_train_pool, 10)
        other_train_final, other_earlystop = held_back(other_train, 20)  # sparser: GLI/MEN pool is large

        train_ids = met_train + other_train_final
        validation_ids = met_earlystop + other_earlystop
        test_ids = met_test + other_val

        entries_out = []
        for split, ids in (("train", train_ids), ("val", validation_ids), ("test", test_ids)):
            for subject_id in ids:
                entries_out.append({**by_id[subject_id], "split": split, "split_reason": "MET_source_holdout_v2"})
        manifest = {
            "entries": entries_out,
            "task": "MET_source_holdout_v2",
            "source_evaluation_split": "val",
            "contains_source_final_test": False,
            "final_test_locked": False,
            "note": (
                "v2: train/val/test include GLI+MEN alongside the held-out MET source branch, unlike "
                "the original loso_met_* fold which was MET-only end to end. See T5.10 correction in "
                "IMPLEMENTATION_LOG.md."
            ),
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        manifest_path = output / f"{name}_split.json"
        atomic_json_dump(manifest, manifest_path)

        fold_config = json.loads(json.dumps(config))  # deep copy
        fold_config["project"]["name"] = f"TumorTrust-{name}"
        fold_config["project"]["output_dir"] = f"outputs/{name}"
        fold_config["project"]["seed"] = 20260813
        fold_config["data"]["split_manifest"] = str(manifest_path)
        fold_config["model"]["joint_weighting"] = "fixed"
        config_path = output / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding="utf-8")
        registry.append(
            {
                "id": name,
                "config": str(config_path),
                "task": "MET_source_holdout_v2",
                "status": "ready",
                "train_subjects": len(train_ids),
                "val_subjects": len(validation_ids),
                "test_subjects": len(test_ids),
            }
        )

    atomic_json_dump(registry, output / "registry_v2.json")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
