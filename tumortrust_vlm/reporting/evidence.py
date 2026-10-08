from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label as connected_components

from tumortrust_vlm.data.constants import REGION_LABELS


@dataclass(frozen=True)
class VolumeEvidence:
    value_ml: float
    interval_90_ml: tuple[float, float] | None = None
    interval_95_ml: tuple[float, float] | None = None


@dataclass(frozen=True)
class EvidenceCard:
    schema_version: str
    subject_id: str
    tumor_family_probabilities: dict[str, float]
    predicted_family: str
    volumes: dict[str, VolumeEvidence]
    component_count: int
    laterality: str
    segmentation_uncertainty: float | None
    classification_uncertainty: float | None
    referral: bool
    referral_reasons: tuple[str, ...]
    unavailable_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> EvidenceCard:
        return cls(
            schema_version=payload["schema_version"],
            subject_id=payload["subject_id"],
            tumor_family_probabilities={
                key: float(value) for key, value in payload["tumor_family_probabilities"].items()
            },
            predicted_family=payload["predicted_family"],
            volumes={
                region: VolumeEvidence(
                    value_ml=float(value["value_ml"]),
                    interval_90_ml=tuple(value["interval_90_ml"])
                    if value.get("interval_90_ml") is not None
                    else None,
                    interval_95_ml=tuple(value["interval_95_ml"])
                    if value.get("interval_95_ml") is not None
                    else None,
                )
                for region, value in payload["volumes"].items()
            },
            component_count=int(payload["component_count"]),
            laterality=payload["laterality"],
            segmentation_uncertainty=payload.get("segmentation_uncertainty"),
            classification_uncertainty=payload.get("classification_uncertainty"),
            referral=bool(payload["referral"]),
            referral_reasons=tuple(payload.get("referral_reasons", ())),
            unavailable_fields=tuple(payload.get("unavailable_fields", ())),
        )


def volumes_from_mask(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    voxel_ml = float(np.prod(spacing)) / 1000.0
    return {
        region: float(np.isin(mask, labels).sum() * voxel_ml)
        for region, labels in REGION_LABELS.items()
    }


def laterality_from_canonical_mask(mask: np.ndarray, midline_fraction: float = 0.05) -> str:
    tumor = mask > 0
    if not tumor.any():
        return "none"
    midline = (mask.shape[0] - 1) / 2
    x = np.where(tumor)[0]
    tolerance = max(1.0, mask.shape[0] * midline_fraction)
    left = np.sum(x < midline - tolerance)
    right = np.sum(x > midline + tolerance)
    central = len(x) - left - right
    if central / len(x) >= 0.25:
        return "midline"
    if left and right and min(left, right) / max(left, right) >= 0.20:
        return "bilateral"
    return "left" if left > right else "right"


def build_evidence_card(
    subject_id: str,
    predicted_mask: np.ndarray,
    spacing: tuple[float, float, float],
    class_probabilities: dict[str, float],
    *,
    volume_intervals_90: dict[str, tuple[float, float]] | None = None,
    volume_intervals_95: dict[str, tuple[float, float]] | None = None,
    segmentation_uncertainty: float | None = None,
    classification_uncertainty: float | None = None,
    referral_threshold: float | None = None,
    segmentation_referral_threshold: float | None = None,
    classification_referral_threshold: float | None = None,
    minimum_component_mm3: float = 100,
    unavailable_fields: tuple[str, ...] = (),
) -> EvidenceCard:
    volume_values = volumes_from_mask(predicted_mask, spacing)
    volumes = {
        region: VolumeEvidence(
            value_ml=value,
            interval_90_ml=(volume_intervals_90 or {}).get(region),
            interval_95_ml=(volume_intervals_95 or {}).get(region),
        )
        for region, value in volume_values.items()
    }
    voxel_volume = float(np.prod(spacing))
    minimum_voxels = max(1, round(minimum_component_mm3 / voxel_volume))
    components, count = connected_components(predicted_mask > 0)
    component_count = sum(
        (components == index).sum() >= minimum_voxels for index in range(1, count + 1)
    )
    reasons = []
    segmentation_threshold = (
        segmentation_referral_threshold
        if segmentation_referral_threshold is not None
        else referral_threshold
    )
    classification_threshold = (
        classification_referral_threshold
        if classification_referral_threshold is not None
        else referral_threshold
    )
    if (
        segmentation_threshold is not None
        and segmentation_uncertainty is not None
        and segmentation_uncertainty > segmentation_threshold
    ):
        reasons.append("segmentation_uncertainty")
    if (
        classification_threshold is not None
        and classification_uncertainty is not None
        and classification_uncertainty > classification_threshold
    ):
        reasons.append("classification_uncertainty")
    predicted_family = max(class_probabilities, key=class_probabilities.get)
    return EvidenceCard(
        schema_version="1.1",
        subject_id=subject_id,
        tumor_family_probabilities=class_probabilities,
        predicted_family=predicted_family,
        volumes=volumes,
        component_count=int(component_count),
        laterality=laterality_from_canonical_mask(predicted_mask),
        segmentation_uncertainty=segmentation_uncertainty,
        classification_uncertainty=classification_uncertainty,
        referral=bool(reasons),
        referral_reasons=tuple(reasons),
        unavailable_fields=tuple(sorted(set(unavailable_fields))),
    )
