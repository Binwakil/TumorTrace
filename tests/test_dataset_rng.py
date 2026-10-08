import random

import numpy as np
import torch

from tumortrust_vlm.data.dataset import TumorTrustDataset
from tumortrust_vlm.engine import (
    _epochs_since_best,
    development_validation_limit,
    stratified_validation_panel,
)
from tumortrust_vlm.utils import capture_rng_state, restore_rng_state


def test_epoch_changes_deterministic_augmentation_stream() -> None:
    dataset = TumorTrustDataset.__new__(TumorTrustDataset)
    dataset.seed = 17
    dataset.epoch = 1
    first = np.random.default_rng(
        np.random.SeedSequence([dataset.seed, dataset.epoch, 4])
    ).random()
    repeated = np.random.default_rng(
        np.random.SeedSequence([dataset.seed, dataset.epoch, 4])
    ).random()
    dataset.set_epoch(2)
    next_epoch = np.random.default_rng(
        np.random.SeedSequence([dataset.seed, dataset.epoch, 4])
    ).random()

    assert first == repeated
    assert first != next_epoch


def test_development_validation_panel_is_deterministic_and_stratified() -> None:
    records = [
        {
            "subject_id": f"{cohort}-{report}-{index}",
            "cohort": cohort,
            "source_branch": "source",
            "has_report": report,
        }
        for cohort in ("GLI", "MEN", "MET")
        for report in (False, True)
        for index in range(5)
    ]
    first = stratified_validation_panel(records, 12, seed=7)
    second = stratified_validation_panel(records, 12, seed=7)

    assert [record["subject_id"] for record in first] == [
        record["subject_id"] for record in second
    ]
    assert {(record["cohort"], record["has_report"]) for record in first} == {
        (cohort, report)
        for cohort in ("GLI", "MEN", "MET")
        for report in (False, True)
    }


def test_classification_uses_full_validation_for_checkpoint_selection() -> None:
    config = {
        "model": {"joint_weighting": "classification_only"},
        "training": {
            "max_validation_cases": None,
            "full_validation_during_training": False,
        },
    }
    assert development_validation_limit(config) is None

    config["model"]["joint_weighting"] = "fixed"
    assert development_validation_limit(config) == 24

    config["training"]["max_validation_cases"] = 12
    assert development_validation_limit(config) == 12


def test_process_rng_streams_can_be_restored_for_exact_resume() -> None:
    original = capture_rng_state()
    try:
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        checkpoint = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.rand(1).item())
        random.random()
        np.random.random()
        torch.rand(4)

        restore_rng_state(checkpoint)
        actual = (random.random(), np.random.random(), torch.rand(1).item())
        assert actual == expected
    finally:
        restore_rng_state(original)


def test_rng_restore_accepts_portable_list_states() -> None:
    original = capture_rng_state()
    try:
        torch.manual_seed(23)
        checkpoint = capture_rng_state()
        portable = {
            **checkpoint,
            "torch_cpu": checkpoint["torch_cpu"].tolist(),
            "torch_cuda": [state.tolist() for state in checkpoint.get("torch_cuda", [])],
        }
        expected = torch.rand(3)

        restore_rng_state(portable)

        assert torch.equal(torch.rand(3), expected)
    finally:
        restore_rng_state(original)


def test_rng_restore_moves_device_mapped_cpu_state_back_to_cpu() -> None:
    if not torch.cuda.is_available():
        return
    original = capture_rng_state()
    try:
        torch.manual_seed(29)
        checkpoint = capture_rng_state()
        checkpoint["torch_cpu"] = checkpoint["torch_cpu"].to("cuda")
        expected = torch.rand(3)

        restore_rng_state(checkpoint)

        assert torch.equal(torch.rand(3), expected)
    finally:
        restore_rng_state(original)


def test_legacy_resume_recovers_early_stopping_progress() -> None:
    history = [
        {"epoch": 5, "val_macro_dice": 0.4},
        {"epoch": 10, "val_macro_dice": 0.6},
        {"epoch": 11},
        {"epoch": 15, "val_macro_dice": 0.5},
        {"epoch": 16},
    ]
    assert _epochs_since_best(history, "macro_dice") == 6
