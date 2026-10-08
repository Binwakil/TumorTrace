import pytest

from scripts.aggregate_results import summarize_runs


def test_multi_seed_summary_has_paired_cohort_intervals() -> None:
    runs = [
        {
            "a": {"subject_id": "a", "cohort": "GLI", "macro_dice": 0.8, "dice_WT": 0.9},
            "b": {"subject_id": "b", "cohort": "MEN", "macro_dice": 0.6, "dice_WT": 0.7},
        },
        {
            "a": {"subject_id": "a", "cohort": "GLI", "macro_dice": 0.9, "dice_WT": 0.95},
            "b": {"subject_id": "b", "cohort": "MEN", "macro_dice": 0.4, "dice_WT": 0.65},
        },
    ]

    result = summarize_runs(runs, bootstrap_samples=20)

    assert result["runs"] == 2
    assert result["paired_subjects"] == 2
    assert result["metrics"]["macro_dice"]["seed_sd"] > 0
    assert result["macro_dice_by_cohort"]["GLI"]["mean"] == pytest.approx(0.85)
    assert result["macro_dice_worst_cohort"] == pytest.approx(0.5)
    assert result["macro_dice_maximum_cohort_gap"] == pytest.approx(0.35)


def test_multi_seed_summary_excludes_partially_missing_metrics() -> None:
    runs = [
        {
            "a": {"cohort": "GLI", "macro_dice": 0.8, "optional": 1.0},
            "b": {"cohort": "MEN", "macro_dice": 0.6},
        },
        {
            "a": {"cohort": "GLI", "macro_dice": 0.9, "optional": 2.0},
            "b": {"cohort": "MEN", "macro_dice": 0.5, "optional": 3.0},
        },
    ]

    result = summarize_runs(runs, bootstrap_samples=20)

    assert "optional" not in result["metrics"]
