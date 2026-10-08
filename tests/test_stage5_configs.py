import json
from pathlib import Path

import yaml

from scripts.materialize_stage5_configs import EXPERIMENTS, materialize_stage5_configs


def test_stage5_configs_change_only_joint_weighting(tmp_path: Path) -> None:
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "project": {
                    "name": "selected",
                    "output_dir": "outputs/selected",
                    "seed": 7,
                },
                "data": {"normalization": "zscore_nonzero"},
                "model": {
                    "joint_weighting": "segmentation_only",
                    "lambda_cls": 0.2,
                },
            }
        )
    )

    registry = materialize_stage5_configs(selected, tmp_path / "stage5")

    assert registry["locked_test_opened"] is False
    assert len(registry["experiments"]) == len(EXPERIMENTS)
    for experiment_id, mode in EXPERIMENTS.items():
        config = yaml.safe_load((tmp_path / "stage5" / f"{experiment_id}.yaml").read_text())
        assert config["data"]["normalization"] == "zscore_nonzero"
        assert config["model"]["joint_weighting"] == mode
        assert config["project"]["seed"] == 7
        assert config["project"]["output_dir"] == f"outputs/{experiment_id}"


def test_stage5_configs_reject_non_segmentation_reference(tmp_path: Path) -> None:
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "project": {"name": "joint", "output_dir": "outputs/joint"},
                "model": {"joint_weighting": "fixed"},
            }
        )
    )

    try:
        materialize_stage5_configs(selected, tmp_path / "stage5")
    except ValueError as error:
        assert "segmentation-only" in str(error)
    else:
        raise AssertionError("Expected a non-segmentation reference to be rejected")
