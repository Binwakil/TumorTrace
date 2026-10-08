from pathlib import Path

import pytest

from scripts.evaluate_nnunet import validate_prediction_set


def test_nnunet_prediction_set_requires_exact_validation_ids(tmp_path: Path) -> None:
    records = [{"case_id": "TT_A"}, {"case_id": "TT_B"}]
    (tmp_path / "TT_A.nii.gz").touch()
    (tmp_path / "TT_B.nii.gz").touch()

    audit = validate_prediction_set(tmp_path, records)

    assert audit == {
        "expected_predictions": 2,
        "observed_predictions": 2,
        "prediction_set_exact": True,
    }


@pytest.mark.parametrize("unexpected", [False, True])
def test_nnunet_prediction_set_rejects_missing_or_unexpected(
    tmp_path: Path, unexpected: bool
) -> None:
    records = [{"case_id": "TT_A"}, {"case_id": "TT_B"}]
    (tmp_path / "TT_A.nii.gz").touch()
    if unexpected:
        (tmp_path / "TT_B.nii.gz").touch()
        (tmp_path / "LOCKED.nii.gz").touch()

    with pytest.raises(ValueError, match="does not match frozen validation cases"):
        validate_prediction_set(tmp_path, records)


def test_nnunet_prediction_set_allows_extra_files_for_limited_smoke(tmp_path: Path) -> None:
    records = [{"case_id": "TT_A"}]
    (tmp_path / "TT_A.nii.gz").touch()
    (tmp_path / "TT_B.nii.gz").touch()

    audit = validate_prediction_set(tmp_path, records, require_exact=False)

    assert audit["prediction_set_exact"] is False
