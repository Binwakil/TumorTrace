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

from tumortrust_vlm.ablation import changed_config_fields
from tumortrust_vlm.config import load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

EXPERIMENTS = {
    "C0": "classification_only",
    "M0": "fixed",
    "M1": "pcgrad",
    "M1_uncertainty_weighting": "uncertainty",
}


def materialize_stage5_configs(selected_config: Path, output: Path) -> dict:
    baseline = load_config(selected_config)
    if baseline["model"]["joint_weighting"] != "segmentation_only":
        raise ValueError("Selected stage-5 reference must be a segmentation-only configuration")

    output.mkdir(parents=True, exist_ok=True)
    experiments = []
    for experiment_id, mode in EXPERIMENTS.items():
        config = copy.deepcopy(baseline)
        config["project"]["name"] = f"TumorTrust-VLM-{experiment_id}"
        config["project"]["output_dir"] = f"outputs/{experiment_id}"
        config["project"]["operation"] = "train"
        config["project"]["trigger"] = None
        config["model"]["joint_weighting"] = mode
        changed = changed_config_fields(config, baseline)
        if changed != {"model.joint_weighting"}:
            raise ValueError(
                f"{experiment_id} is confounded against selected segmentation reference: "
                f"{sorted(changed)}"
            )
        path = output / f"{experiment_id}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        experiments.append(
            {
                "id": experiment_id,
                "mode": mode,
                "config": str(path),
                "config_sha256": sha256_file(path),
                "changed_fields_vs_selected_reference": sorted(changed),
            }
        )

    registry = {
        "schema_version": 1,
        "selected_segmentation_config": str(selected_config),
        "selected_segmentation_config_sha256": sha256_file(selected_config),
        "locked_test_opened": False,
        "experiments": experiments,
    }
    atomic_json_dump(registry, output / "registry.json")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize stage-5 configs from the frozen best SegResNet configuration."
    )
    parser.add_argument(
        "--selected-config", default="outputs/D3_zscore/resolved_config.json"
    )
    parser.add_argument("--output", default="artifacts/private/stage5_configs")
    args = parser.parse_args()
    registry = materialize_stage5_configs(Path(args.selected_config), Path(args.output))
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
