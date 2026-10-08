import json
import sys
from pathlib import Path

import pytest

from tumortrust_vlm.nnunet_finalize import (
    evaluation_complete,
    expected_predictions,
    prediction_set_exact,
    validation_command,
)


def test_best_validation_command_is_validation_only() -> None:
    final = validation_command(best=False)
    best = validation_command(best=True)

    assert "--val" in final
    assert "--val_best" not in final
    assert "--val" in best
    assert "--val_best" in best
    assert "--c" not in best
    assert best[:3] == [sys.executable, "-m", "nnunetv2.run.run_training"]


def test_expected_predictions_rejects_locked_test_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            [
                {"case_id": "A", "subject_id": "a", "split": "val"},
                {"case_id": "B", "subject_id": "b", "split": "test"},
            ]
        )
    )

    with pytest.raises(ValueError, match="locked test"):
        expected_predictions(mapping)


def test_prediction_and_evaluation_completion_require_exact_subjects(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "A.nii.gz").touch()
    (predictions / "B.nii.gz").touch()
    assert prediction_set_exact(predictions, {"A.nii.gz", "B.nii.gz"}) is True

    cases = tmp_path / "cases.json"
    summary = tmp_path / "summary.json"
    cases.write_text(json.dumps([{"subject_id": "a"}, {"subject_id": "b"}]))
    summary.write_text(
        json.dumps(
            {
                "prediction_audit": {"prediction_set_exact": True},
                "locked_test_opened": False,
            }
        )
    )
    assert evaluation_complete(cases, summary, {"a", "b"}) is True

    cases.write_text(json.dumps([{"subject_id": "a"}, {"subject_id": "locked"}]))
    with pytest.raises(ValueError, match="subject mismatch"):
        evaluation_complete(cases, summary, {"a", "b"})
