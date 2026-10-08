from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from tumortrust_vlm.data.constants import MODALITIES


@dataclass
class SpatialTransform:
    original_affine: np.ndarray
    original_shape: tuple[int, int, int]
    canonical_affine: np.ndarray
    canonical_shape: tuple[int, int, int]
    crop_slices: tuple[slice, slice, slice]
    pre_crop_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]


def normalize_nonzero(array: np.ndarray, method: str = "robust_nonzero") -> np.ndarray:
    result = array.astype(np.float32, copy=True)
    mask = result != 0
    if not mask.any():
        return np.zeros_like(result, dtype=np.float32)
    values = result[mask]
    if method == "robust_nonzero":
        low, high = np.percentile(values, (0.5, 99.5))
        clipped = np.clip(values, low, high)
        median = float(np.median(clipped))
        q1, q3 = np.percentile(clipped, (25, 75))
        scale = max(float(q3 - q1), 1e-6)
        result[mask] = (clipped - median) / scale
    elif method == "zscore_nonzero":
        mean = float(values.mean())
        scale = max(float(values.std()), 1e-6)
        result[mask] = (values - mean) / scale
    else:
        raise ValueError(f"Unknown normalization: {method}")
    result[~mask] = 0
    return result


def brain_bounding_box(
    image: np.ndarray, margin: int | None = 8
) -> tuple[slice, slice, slice]:
    if image.ndim != 4:
        raise ValueError(f"Expected [C,X,Y,Z], got {image.shape}")
    if margin is None:
        return tuple(slice(0, size) for size in image.shape[1:])  # type: ignore[return-value]
    if margin < 0:
        raise ValueError("Brain-crop margin cannot be negative")
    support = np.any(image != 0, axis=0)
    if not support.any():
        return tuple(slice(0, size) for size in image.shape[1:])  # type: ignore[return-value]
    coordinates = np.where(support)
    slices = []
    for axis, size in enumerate(support.shape):
        start = max(0, int(coordinates[axis].min()) - margin)
        stop = min(size, int(coordinates[axis].max()) + margin + 1)
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def _resample_array(array: np.ndarray, old_spacing: Sequence[float], new_spacing: Sequence[float], order: int) -> np.ndarray:
    factors = tuple(float(old) / float(new) for old, new in zip(old_spacing, new_spacing))
    if np.allclose(factors, 1.0, atol=1e-5):
        return array
    return zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)


def load_preprocessed_case(
    record: dict,
    *,
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    normalization: str = "robust_nonzero",
    crop_margin: int | None = 8,
    require_segmentation: bool = False,
    canonicalize: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, SpatialTransform, np.ndarray]:
    images: list[np.ndarray] = []
    canonical_niftis: list[nib.Nifti1Image] = []
    for modality in MODALITIES:
        path = record["sequences"].get(modality)
        if not path:
            raise FileNotFoundError(f"{record['subject_id']} missing {modality}")
        source = nib.load(path)
        current = nib.as_closest_canonical(source) if canonicalize else source
        canonical_niftis.append(current)
        images.append(np.asarray(current.dataobj, dtype=np.float32))

    reference = canonical_niftis[0]
    for image in canonical_niftis[1:]:
        if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"Misaligned sequences for {record['subject_id']}")
    original_reference = nib.load(record["sequences"][MODALITIES[0]])
    old_spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    stack = np.stack(
        [normalize_nonzero(_resample_array(value, old_spacing, target_spacing, order=1), normalization) for value in images]
    )
    presence = np.ones(len(MODALITIES), dtype=np.float32)

    segmentation = None
    if record.get("segmentation"):
        source_seg = nib.load(record["segmentation"])
        current_seg = nib.as_closest_canonical(source_seg) if canonicalize else source_seg
        if current_seg.shape[:3] != reference.shape[:3] or not np.allclose(current_seg.affine, reference.affine, atol=1e-4):
            raise ValueError(f"Image/mask misalignment for {record['subject_id']}")
        segmentation = _resample_array(
            np.asarray(current_seg.dataobj), old_spacing, target_spacing, order=0
        ).astype(np.int16)
        segmentation[segmentation == 4] = 3
    elif require_segmentation:
        raise FileNotFoundError(f"No segmentation for {record['subject_id']}")

    crop = brain_bounding_box(stack, margin=crop_margin)
    pre_crop_shape = tuple(int(value) for value in stack.shape[1:])
    stack = stack[(slice(None),) + crop]
    if segmentation is not None:
        segmentation = segmentation[crop]
    transform = SpatialTransform(
        original_affine=np.asarray(original_reference.affine),
        original_shape=tuple(int(value) for value in original_reference.shape[:3]),
        canonical_affine=np.asarray(reference.affine),
        canonical_shape=tuple(int(value) for value in reference.shape[:3]),
        crop_slices=crop,
        pre_crop_shape=pre_crop_shape,
        spacing=tuple(float(value) for value in target_spacing),
    )
    return stack, segmentation, transform, presence


def restore_crop(prediction: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    restored = np.zeros(transform.pre_crop_shape, dtype=prediction.dtype)
    restored[transform.crop_slices] = prediction
    return restored


def _fit_spatial_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    fitted = np.zeros(shape, dtype=array.dtype)
    shared = tuple(slice(0, min(current, target)) for current, target in zip(array.shape, shape))
    fitted[shared] = array[shared]
    return fitted


def restore_to_source(prediction: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    """Invert crop, spacing resampling, and canonical orientation for a label map."""
    restored = restore_crop(prediction, transform)
    if restored.shape != transform.canonical_shape:
        factors = tuple(
            target / current
            for current, target in zip(restored.shape, transform.canonical_shape)
        )
        restored = zoom(restored, factors, order=0, mode="nearest", prefilter=False)
        restored = _fit_spatial_shape(restored, transform.canonical_shape)
    canonical_orientation = nib.orientations.io_orientation(transform.canonical_affine)
    original_orientation = nib.orientations.io_orientation(transform.original_affine)
    inverse_orientation = nib.orientations.ornt_transform(
        canonical_orientation, original_orientation
    )
    restored = nib.orientations.apply_orientation(restored, inverse_orientation)
    return _fit_spatial_shape(restored, transform.original_shape)


def extract_patch(
    image: np.ndarray,
    segmentation: np.ndarray,
    patch_size: Sequence[int],
    rng: np.random.Generator,
    positive_probability: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    spatial = image.shape[1:]
    target = tuple(int(value) for value in patch_size)
    padded_shape = tuple(max(current, desired) for current, desired in zip(spatial, target))
    if padded_shape != spatial:
        image_pad = [(0, 0)] + [(0, desired - current) for current, desired in zip(spatial, padded_shape)]
        label_pad = [(0, desired - current) for current, desired in zip(spatial, padded_shape)]
        image = np.pad(image, image_pad)
        segmentation = np.pad(segmentation, label_pad)
        spatial = padded_shape
    foreground = np.argwhere(segmentation > 0)
    if foreground.size and rng.random() < positive_probability:
        center = foreground[rng.integers(0, len(foreground))]
    else:
        center = np.asarray([rng.integers(0, size) for size in spatial])
    starts = [
        min(max(0, int(center[axis]) - target[axis] // 2), spatial[axis] - target[axis])
        for axis in range(3)
    ]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, target))
    return image[(slice(None),) + slices], segmentation[slices]


def apply_input_view(image: np.ndarray, segmentation: np.ndarray | None, view: str) -> np.ndarray:
    if view == "whole" or segmentation is None:
        return image
    tumor = segmentation > 0
    if view == "background_only":
        return image * (~tumor)[None]
    if view == "tumor_only":
        return image * tumor[None]
    raise ValueError(f"Unknown input view: {view}")


def simulate_missing_modalities(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.0,
    force: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = image.copy()
    presence = np.ones(len(MODALITIES), dtype=np.float32)
    if force is not None:
        index = MODALITIES.index(force)
        result[index] = 0
        presence[index] = 0
    elif probability > 0 and rng.random() < probability:
        index = int(rng.integers(0, len(MODALITIES)))
        result[index] = 0
        presence[index] = 0
    return result, presence


def apply_robustness_shift(image: np.ndarray, shift: str, rng: np.random.Generator) -> np.ndarray:
    if shift in ("none", ""):
        return image
    result = image.copy()
    support = np.any(result != 0, axis=0)
    if shift == "gaussian_noise":
        noise = rng.normal(0, 0.10, size=result.shape).astype(np.float32)
        result += noise * support[None]
    elif shift == "intensity_scale":
        scales = rng.uniform(0.75, 1.25, size=(result.shape[0], 1, 1, 1)).astype(np.float32)
        result *= scales
    elif shift == "bias_field":
        coordinates = [np.linspace(-1, 1, size, dtype=np.float32) for size in result.shape[1:]]
        x, y, z = np.meshgrid(*coordinates, indexing="ij")
        field = np.exp(0.25 * (x + 0.5 * y - 0.25 * z)).astype(np.float32)
        result *= field[None]
    elif shift == "lower_resolution":
        down = zoom(result, (1, 0.5, 0.5, 0.5), order=1)
        factors = tuple(original / current for original, current in zip(result.shape, down.shape))
        result = zoom(down, factors, order=1)
        result = result[tuple(slice(0, size) for size in image.shape)]
    else:
        raise ValueError(f"Unknown robustness shift: {shift}")
    result[:, ~support] = 0
    return result.astype(np.float32)
