import json
import math
from pathlib import Path

import pytest

from scripts.freeze_segmentation_baseline import freeze_selection


def write_candidate(path: Path, values: list[tuple[str, str, float, float]]) -> None:
    records = []
    for subject, cohort, macro_dice, hd95 in values:
        record = {
            "subject_id": subject,
            "cohort": cohort,
            "macro_dice": macro_dice,
            "hd95_WT": hd95,
        }
        for metric in (
            "dice_WT",
            "dice_TC",
            "dice_ET",
            "dice_SNFH",
            "surface_dice_WT",
            "surface_dice_TC",
            "surface_dice_ET",
            "surface_dice_SNFH",
            "lesion_f1",
            "small_lesion_recall",
            "laterality_correct",
        ):
            record[metric] = macro_dice
        for metric in (
            "hd95_TC",
            "hd95_ET",
            "hd95_SNFH",
            "relative_volume_error_WT",
            "component_count_absolute_error_WT",
        ):
            record[metric] = hd95
        records.append(record)
    path.write_text(json.dumps(records))


def test_freeze_selection_uses_identical_validation_subjects_and_hashes(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "entries": [
                    {"subject_id": "a", "split": "val"},
                    {"subject_id": "b", "split": "val"},
                    {"subject_id": "locked", "split": "test"},
                ]
            }
        )
    )
    strong = tmp_path / "strong.json"
    baseline = tmp_path / "baseline.json"
    write_candidate(strong, [("a", "GLI", 0.9, 4.0), ("b", "MEN", 0.8, 5.0)])
    write_candidate(baseline, [("a", "GLI", 0.8, 3.8), ("b", "MEN", 0.7, 4.8)])
    artifacts = []
    for name, cases in (("strong", strong), ("baseline", baseline)):
        config = tmp_path / f"{name}.yaml"
        checkpoint = tmp_path / f"{name}.pt"
        config.write_text(name)
        checkpoint.write_bytes(name.encode())
        artifacts.append((name, cases, config, checkpoint))

    result = freeze_selection(artifacts, split, bootstrap_samples=50, hd95_tolerance_mm=0.5)

    assert result["selected_candidate"] == "strong"
    assert result["paired_subjects"] == 2
    assert result["locked_test_opened"] is False
    assert result["primary_superiority_supported"] is True
    assert result["wt_hd95_within_tolerance"] is True
    assert len(result["provenance"]["strong"]["checkpoint_sha256"]) == 64


def test_freeze_selection_rejects_missing_or_locked_subjects(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "entries": [
                    {"subject_id": "a", "split": "val"},
                    {"subject_id": "locked", "split": "test"},
                ]
            }
        )
    )
    cases = tmp_path / "cases.json"
    write_candidate(cases, [("locked", "GLI", 0.9, 4.0)])
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("config")
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="does not exactly match frozen validation subjects"):
        freeze_selection(
            [("first", cases, config, checkpoint), ("second", cases, config, checkpoint)],
            split,
            bootstrap_samples=10,
            hd95_tolerance_mm=0.5,
        )


def test_freeze_selection_uses_paired_finite_subset_for_undefined_metrics(
    tmp_path: Path,
) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "entries": [
                    {"subject_id": "a", "split": "val"},
                    {"subject_id": "b", "split": "val"},
                ]
            }
        )
    )
    strong = tmp_path / "strong.json"
    baseline = tmp_path / "baseline.json"
    write_candidate(strong, [("a", "GLI", 0.9, 4.0), ("b", "MEN", 0.8, math.nan)])
    write_candidate(baseline, [("a", "GLI", 0.8, 4.2), ("b", "MEN", 0.7, 5.0)])
    artifacts = []
    for name, cases in (("strong", strong), ("baseline", baseline)):
        config = tmp_path / f"{name}.yaml"
        checkpoint = tmp_path / f"{name}.pt"
        config.write_text(name)
        checkpoint.write_bytes(name.encode())
        artifacts.append((name, cases, config, checkpoint))

    result = freeze_selection(artifacts, split, bootstrap_samples=20, hd95_tolerance_mm=0.5)

    comparison = result["paired_comparisons"]["strong_vs_baseline"]["hd95_WT"]
    assert comparison["paired_n"] == 1
    assert result["candidates"]["strong"]["effective_n"]["hd95_WT"] == 1
