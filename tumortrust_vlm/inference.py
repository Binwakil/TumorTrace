from __future__ import annotations

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.nn import functional as F

from tumortrust_vlm.models.core import MultiTaskSegResNet


def pad_to_multiple(
    image: torch.Tensor, multiple: int = 16
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    original = tuple(int(value) for value in image.shape[-3:])
    padding = [(multiple - size % multiple) % multiple for size in original]
    padded = F.pad(image, (0, padding[2], 0, padding[1], 0, padding[0]))
    return padded, original


def crop_to_shape(array: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return array[..., : shape[0], : shape[1], : shape[2]]


def infer_core_once(
    model: MultiTaskSegResNet,
    image: torch.Tensor,
    modality_presence: torch.Tensor,
    *,
    roi_size: tuple[int, int, int],
    sw_batch_size: int = 1,
    overlap: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer all core outputs without using a full-volume decoder for large inputs."""
    fits_roi = all(image.shape[axis + 2] <= roi_size[axis] for axis in range(3))
    if fits_roi:
        output = model(image, modality_presence)
        return output["segmentation"], output["classification"], output["latent"]

    segmentation_logits = sliding_window_inference(
        image,
        roi_size,
        sw_batch_size,
        lambda patch: model.segment(patch)[0],
        overlap=overlap,
    )
    reporting_latent, classification_latent = model.encode_for_reporting_and_classification(image)
    classification_logits = model.classify_latent(
        classification_latent, modality_presence, image.shape[1]
    )
    return segmentation_logits, classification_logits, reporting_latent


def evidence_vector(card: dict) -> list[float]:
    probabilities = card["tumor_family_probabilities"]
    vector = [float(probabilities[name]) for name in ("GLI", "MEN", "MET")]
    vector.extend(
        float(np.log1p(card["volumes"][region]["value_ml"]))
        for region in ("WT", "TC", "ET", "SNFH")
    )
    laterality = card["laterality"]
    vector.extend(
        float(laterality == value) for value in ("left", "right", "bilateral", "midline", "none")
    )
    vector.append(float(np.log1p(card["component_count"])))
    vector.append(float(card["segmentation_uncertainty"] or 0))
    vector.append(float(card["classification_uncertainty"] or 0))
    vector.append(float(card["referral"]))
    return vector
