#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from tumortrust_vlm.config import config_hash, load_config
from tumortrust_vlm.data.dataset import collate_training
from tumortrust_vlm.data.splits import split_requires_final_unlock
from tumortrust_vlm.engine import build_dataset, build_model, evaluate_model, load_checkpoint
from tumortrust_vlm.utils import atomic_json_dump, sha256_file, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--device")
    parser.add_argument("--export-official")
    parser.add_argument("--case-output")
    args = parser.parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    config = load_config(config_path)
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split_requires_final_unlock(manifest, args.split) and not args.unlock_final_test:
        raise SystemExit(
            "Final test is locked. Freeze the protocol, then pass --unlock-final-test once."
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    dataset = build_dataset(config, args.split, False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_training
    )
    aggregate, cases = evaluate_model(
        model,
        loader,
        config,
        device,
        max_cases=args.max_cases,
        official_export_dir=args.export_official,
    )
    evaluated_subjects = sorted(case["subject_id"] for case in cases)
    aggregate["evaluation_provenance"] = {
        "split": args.split,
        "subjects": len(evaluated_subjects),
        "subject_ids_sha256": sha256_json(evaluated_subjects),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_hash": config_hash(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "locked_test_opened": bool(args.split == "test" and args.unlock_final_test),
    }
    output = Path(config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(aggregate, output / f"{args.split}_aggregate.json")
    case_output = (
        Path(args.case_output) if args.case_output else output / f"{args.split}_cases.json"
    )
    atomic_json_dump(cases, case_output)
    print(json.dumps({"checkpoint_epoch": checkpoint.get("epoch"), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
