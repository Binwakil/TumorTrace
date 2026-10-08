import json
from pathlib import Path

import pytest

from tumortrust_vlm.config import config_hash
from tumortrust_vlm.experiment_queue import evaluation_complete, next_action, training_complete
from tumortrust_vlm.utils import sha256_file


def setup_experiment(tmp_path: Path) -> tuple[Path, dict, Path]:
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {"subject_id": "train", "split": "train"},
                    {"subject_id": "val", "split": "val"},
                    {"subject_id": "locked", "split": "test"},
                ]
            }
        )
    )
    output = tmp_path / "run"
    config = {
        "project": {"output_dir": str(output)},
        "data": {"split_manifest": str(manifest)},
        "model": {"joint_weighting": "classification_only"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, config, output


def complete_training(config_path: Path, config: dict, output: Path) -> Path:
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"best")
    (output / "resolved_config.json").write_text(json.dumps(config))
    (output / "summary.json").write_text(json.dumps({"config_hash": config_hash(config)}))
    return checkpoint


def test_queue_actions_advance_from_train_to_evaluate_to_complete(tmp_path: Path) -> None:
    config_path, config, output = setup_experiment(tmp_path)
    assert next_action(config_path) == ("train", None)

    (output / "checkpoints").mkdir(parents=True)
    last = output / "checkpoints" / "last.pt"
    last.write_bytes(b"last")
    assert next_action(config_path) == ("resume_train", last)

    checkpoint = complete_training(config_path, config, output)
    assert training_complete(config_path) is True
    assert next_action(config_path) == ("evaluate", checkpoint)

    (output / "val_cases.json").write_text(json.dumps([{"subject_id": "val"}]))
    (output / "val_aggregate.json").write_text(
        json.dumps(
            {
                "evaluation_provenance": {
                    "split": "val",
                    "locked_test_opened": False,
                    "config_hash": config_hash(config),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            }
        )
    )
    assert evaluation_complete(config_path) is True
    assert next_action(config_path) == ("complete", None)


def test_queue_rejects_validation_subject_mismatch(tmp_path: Path) -> None:
    config_path, config, output = setup_experiment(tmp_path)
    checkpoint = complete_training(config_path, config, output)
    (output / "val_cases.json").write_text(json.dumps([{"subject_id": "locked"}]))
    (output / "val_aggregate.json").write_text(
        json.dumps(
            {
                "evaluation_provenance": {
                    "split": "val",
                    "locked_test_opened": False,
                    "config_hash": config_hash(config),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Validation case set mismatch"):
        evaluation_complete(config_path)
