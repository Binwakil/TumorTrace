import json
from pathlib import Path

from tumortrust_vlm.monitor import (
    d3_status,
    format_duration,
    nnunet_status,
    progress_bar,
    stage5_status,
)


def test_d3_status_reads_latest_and_best_validation(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    config = tmp_path / "config.json"
    history.write_text(
        '[{"epoch": 1, "loss": 1.0}, '
        '{"epoch": 10, "loss": 0.7, "val_macro_dice": 0.6}, '
        '{"epoch": 20, "loss": 0.5, "val_macro_dice": 0.55}]'
    )
    config.write_text('{"training": {"epochs": 100, "validate_every": 10, "patience": 50}}')

    result = d3_status(history, config)

    assert result["epoch"] == 20
    assert result["latest_macro_dice"] == 0.55
    assert result["best_macro_dice"] == 0.6
    assert result["best_validation_epoch"] == 10


def test_nnunet_status_uses_latest_log_and_global_best(tmp_path: Path) -> None:
    first = tmp_path / "training_log_first.txt"
    resumed = tmp_path / "training_log_resumed.txt"
    first.write_text(
        "2026-08-10 12:00:00.000000: Epoch 800\n"
        "2026-08-10 12:01:16.000000: Yayy! New best EMA pseudo Dice: 0.88\n"
    )
    resumed.write_text(
        "2026-08-11 12:00:00.000000: Epoch 850\n"
        "2026-08-11 12:01:17.000000: Current learning rate: 0.0018\n"
        "2026-08-11 12:01:17.000000: Pseudo dice [0.8, 0.9, 0.95]\n"
        "2026-08-11 12:01:17.000000: Epoch time: 77.0 s\n"
    )
    first.touch()
    resumed.touch()

    result = nnunet_status([first, resumed])

    assert result["epoch"] == 850
    assert result["pseudo_dice"] == [0.8, 0.9, 0.95]
    assert result["best_ema_pseudo_dice"] == 0.88
    assert result["epoch_seconds"] == 77.0


def test_monitor_formatting_helpers() -> None:
    assert progress_bar(50, 100, width=10) == "[#####-----]  50.0%"
    assert format_duration(3665) == "1h 01m"
    assert format_duration(None) == "learning..."


def test_stage5_status_tracks_current_and_completed_experiments(tmp_path: Path) -> None:
    config_root = tmp_path / "artifacts/private/stage5_configs"
    config_root.mkdir(parents=True)
    experiments = []
    for experiment_id, mode in (("C0", "classification_only"), ("M0", "fixed")):
        config = config_root / f"{experiment_id}.yaml"
        config.write_text(
            "\n".join(
                (
                    "project:",
                    f"  output_dir: outputs/{experiment_id}",
                    "model:",
                    f"  joint_weighting: {mode}",
                    "training:",
                    "  epochs: 300",
                )
            )
        )
        experiments.append({"id": experiment_id, "config": str(config.relative_to(tmp_path))})
    registry = config_root / "registry.json"
    registry.write_text(json.dumps({"experiments": experiments}))
    queue = tmp_path / "artifacts/private/stage5_queue_status.json"
    queue.write_text(
        json.dumps(
            {
                "experiments": [
                    {"id": "C0", "action": "complete"},
                    {"id": "M0", "action": "train"},
                ]
            }
        )
    )
    c0 = tmp_path / "outputs/C0"
    c0.mkdir(parents=True)
    (c0 / "summary.json").write_text('{"best_validation_balanced_accuracy": 0.8}')
    (c0 / "val_aggregate.json").write_text("{}")
    m0 = tmp_path / "outputs/M0"
    m0.mkdir(parents=True)
    (m0 / "history.json").write_text(
        '[{"epoch": 10, "val_macro_dice": 0.7}]'
    )

    result = stage5_status(tmp_path, registry, queue)

    assert result["completed_experiments"] == 1
    assert result["total_experiments"] == 2
    assert result["current"]["id"] == "M0"
    assert result["current"]["action"] == "train"
    assert result["current"]["latest_validation"] == 0.7
    assert result["current"]["best_validation"] == 0.7
