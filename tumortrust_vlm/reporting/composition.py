from __future__ import annotations

import copy

from tumortrust_vlm.inference import evidence_vector


def _index(records: list[dict], recipe: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        subject_id = record["subject_id"]
        if subject_id in indexed:
            raise ValueError(f"Duplicate {recipe} feature record: {subject_id}")
        if not record.get("subject_provenance", {}).get(
            "excluded_from_checkpoint_training", False
        ):
            raise ValueError(f"{recipe} feature record is not out-of-fold: {subject_id}")
        indexed[subject_id] = record
    return indexed


def compose_reporter_features(
    segmentation_records: list[dict], classification_records: list[dict]
) -> list[dict]:
    """Compose reporter inputs from matched single-task OOF recipes.

    Segmentation-derived masks, burden fields, uncertainty, and region-aware visual
    tokens come from the frozen segmentation-only recipe. Tumor-family probabilities
    and classification uncertainty come from the classification-only recipe. Both
    records must prove that the subject was excluded from checkpoint training.
    """

    segmentation = _index(segmentation_records, "segmentation")
    classification = _index(classification_records, "classification")
    if segmentation.keys() != classification.keys():
        missing_classification = sorted(segmentation.keys() - classification.keys())
        missing_segmentation = sorted(classification.keys() - segmentation.keys())
        raise ValueError(
            "OOF recipe subject sets differ: "
            f"missing_classification={missing_classification[:5]}, "
            f"missing_segmentation={missing_segmentation[:5]}"
        )

    composed = []
    for subject_id in sorted(segmentation):
        seg = segmentation[subject_id]
        cls = classification[subject_id]
        for field in ("cohort", "source_branch"):
            if seg.get(field) != cls.get(field):
                raise ValueError(f"OOF recipe {field} mismatch for {subject_id}")

        record = copy.deepcopy(seg)
        evidence = record["evidence"]
        cls_evidence = cls["evidence"]
        probabilities = {
            name: float(cls_evidence["tumor_family_probabilities"][name])
            for name in ("GLI", "MEN", "MET")
        }
        evidence["tumor_family_probabilities"] = probabilities
        evidence["predicted_family"] = max(probabilities, key=probabilities.get)
        evidence["classification_uncertainty"] = cls_evidence.get(
            "classification_uncertainty"
        )

        segmentation_reasons = [
            reason
            for reason in evidence.get("referral_reasons", [])
            if reason != "classification_uncertainty"
        ]
        classification_reasons = [
            reason
            for reason in cls_evidence.get("referral_reasons", [])
            if reason == "classification_uncertainty"
        ]
        evidence["referral_reasons"] = sorted(
            set(segmentation_reasons + classification_reasons)
        )
        evidence["referral"] = bool(evidence["referral_reasons"])
        record["evidence_vector"] = evidence_vector(evidence)

        uncertainty = copy.deepcopy(seg.get("uncertainty", {}))
        for key, value in cls.get("uncertainty", {}).items():
            if key.startswith("classification_"):
                uncertainty[key] = value
        record["uncertainty"] = uncertainty
        record["inference_provenance"] = {
            "composition": "segmentation_only_plus_classification_only_oof_v1",
            "segmentation_recipe": seg.get("inference_provenance", {}),
            "classification_recipe": cls.get("inference_provenance", {}),
        }
        record["subject_provenance"] = {
            "feature_extraction_split": seg["subject_provenance"].get(
                "feature_extraction_split"
            ),
            "excluded_from_checkpoint_training": True,
            "segmentation_recipe_excluded_from_training": True,
            "classification_recipe_excluded_from_training": True,
        }
        composed.append(record)
    return composed
