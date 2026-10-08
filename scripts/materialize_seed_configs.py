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
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/private/seed_configs")
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260810, 20260811, 20260812))
    parser.add_argument(
        "--mode",
        choices=("segmentation_only", "classification_only", "fixed", "pcgrad", "uncertainty"),
        default="segmentation_only",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=10)
    args = parser.parse_args()
    base = load_config(args.config)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    registry = []
    for index, seed in enumerate(args.seeds):
        config = copy.deepcopy(base)
        run_id = f"{args.mode}_seed{index}"
        config["project"]["name"] = f"TumorTrust-{run_id}"
        config["project"]["seed"] = seed
        config["project"]["output_dir"] = f"outputs/{run_id}"
        config["model"]["joint_weighting"] = args.mode
        if args.mode == "segmentation_only":
            config["data"]["missing_modality_probability"] = 0.0
        config["training"]["max_steps_per_epoch"] = args.steps_per_epoch
        config["training"]["patch_validation"] = False
        config["training"]["validate_every"] = args.validate_every
        path = output / f"{run_id}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        registry.append({"id": run_id, "seed": seed, "config": str(path)})
    atomic_json_dump(registry, output / f"{args.mode}_registry.json")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
