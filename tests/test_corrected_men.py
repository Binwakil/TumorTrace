import numpy as np

from scripts.audit_corrected_men import compare_labels


def test_corrected_label_audit_quantifies_region_change() -> None:
    original = np.zeros((8, 8, 8), dtype=np.uint8)
    corrected = np.zeros_like(original)
    original[1:3, 1:3, 1:3] = 3
    corrected[1:4, 1:3, 1:3] = 3

    result = compare_labels(original, corrected, voxel_volume_mm3=1.0)

    assert result["changed_voxels"] == 4
    assert result["dice_WT"] == 0.8
    assert result["absolute_volume_change_ml_WT"] == 0.004
    assert not result["exactly_identical"]
