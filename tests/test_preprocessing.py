from pathlib import Path

import nibabel as nib
import numpy as np

from tumortrust_vlm.data.preprocessing import (
    apply_input_view,
    apply_robustness_shift,
    brain_bounding_box,
    extract_patch,
    load_preprocessed_case,
    normalize_nonzero,
    restore_to_source,
)


def write_case(root: Path, orientation: str = "LPS", include_segmentation: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    subject_id = "BraTS-GLI-00001-000"
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) if orientation == "LPS" else np.eye(4)
    base = np.zeros((20, 18, 16), dtype=np.float32)
    base[3:17, 2:16, 2:14] = np.arange(14 * 14 * 12).reshape(14, 14, 12) + 1
    sequences = {}
    for index, modality in enumerate(("t1n", "t1c", "t2w", "t2f"), start=1):
        path = root / f"{subject_id}-{modality}.nii.gz"
        nib.save(nib.Nifti1Image(base * index, affine), path)
        sequences[modality] = str(path)
    segmentation = None
    if include_segmentation:
        mask = np.zeros_like(base, dtype=np.int16)
        mask[8:12, 7:11, 6:10] = 3
        path = root / f"{subject_id}-seg.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), path)
        segmentation = str(path)
    return {"subject_id": subject_id, "sequences": sequences, "segmentation": segmentation}


def test_mask_independent_inference_load(tmp_path):
    record = write_case(tmp_path, include_segmentation=False)
    image, segmentation, transform, presence = load_preprocessed_case(record, crop_margin=1)
    assert segmentation is None
    assert image.shape[0] == 4
    assert image.shape[1:] == transform.pre_crop_shape or all(
        image.shape[index + 1] <= transform.pre_crop_shape[index] for index in range(3)
    )
    assert presence.tolist() == [1, 1, 1, 1]


def test_lps_is_canonicalized_with_mask_alignment(tmp_path):
    record = write_case(tmp_path, orientation="LPS")
    image, segmentation, transform, _ = load_preprocessed_case(record, crop_margin=1, require_segmentation=True)
    assert segmentation is not None
    assert image.shape[1:] == segmentation.shape
    assert nib.aff2axcodes(transform.canonical_affine) == ("R", "A", "S")
    assert segmentation.max() == 3

    source_prediction = restore_to_source(segmentation, transform)
    source = nib.load(record["segmentation"])
    canonical_round_trip = np.asarray(
        nib.as_closest_canonical(
            nib.Nifti1Image(source_prediction, source.affine)
        ).dataobj
    )
    uncropped = np.zeros(transform.pre_crop_shape, dtype=segmentation.dtype)
    uncropped[transform.crop_slices] = segmentation
    assert np.array_equal(canonical_round_trip, uncropped)


def test_normalization_and_views_and_patch():
    image = np.zeros((4, 20, 20, 20), dtype=np.float32)
    image[:, 3:17, 3:17, 3:17] = 5
    mask = np.zeros((20, 20, 20), dtype=np.int16)
    mask[8:12, 8:12, 8:12] = 3
    assert np.isfinite(normalize_nonzero(image[0])).all()
    box = brain_bounding_box(image, margin=1)
    assert box[0].start == 2 and box[0].stop == 18
    background = apply_input_view(image, mask, "background_only")
    tumor = apply_input_view(image, mask, "tumor_only")
    assert np.all(background[:, mask > 0] == 0)
    assert np.all(tumor[:, mask == 0] == 0)
    patch_image, patch_mask = extract_patch(image, mask, (12, 12, 12), np.random.default_rng(0), 1.0)
    assert patch_image.shape == (4, 12, 12, 12)
    assert patch_mask.shape == (12, 12, 12)
    assert patch_mask.any()
    shifted = apply_robustness_shift(image, "gaussian_noise", np.random.default_rng(1))
    assert shifted.shape == image.shape
    assert np.all(shifted[:, np.all(image == 0, axis=0)] == 0)


def test_disabled_brain_crop_preserves_full_volume() -> None:
    image = np.zeros((4, 12, 14, 16), dtype=np.float32)
    image[:, 4:8, 5:9, 6:10] = 1
    crop = brain_bounding_box(image, margin=None)
    assert image[(slice(None),) + crop].shape == image.shape
