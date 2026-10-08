from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn.metrics import f1_score

from tumortrust_vlm.reporting.evidence import laterality_from_canonical_mask, volumes_from_mask


def concordance_correlation_coefficient(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    covariance = np.mean((predicted - predicted.mean()) * (target - target.mean()))
    denominator = predicted.var() + target.var() + (predicted.mean() - target.mean()) ** 2
    return float(2 * covariance / denominator) if denominator else 1.0


def case_volume_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    regions: tuple[str, ...] = ("WT", "TC", "ET", "SNFH"),
    minimum_component_mm3: float = 100.0,
) -> dict[str, float | str]:
    predicted = volumes_from_mask(prediction, spacing)
    truth = volumes_from_mask(target, spacing)
    metrics: dict[str, float | str] = {}
    for region in regions:
        absolute = abs(predicted[region] - truth[region])
        target_present = truth[region] > 0
        predicted_present = predicted[region] > 0
        relative = absolute / truth[region] if target_present else float("nan")
        metrics[f"predicted_volume_ml_{region}"] = predicted[region]
        metrics[f"target_volume_ml_{region}"] = truth[region]
        metrics[f"absolute_volume_error_ml_{region}"] = absolute
        metrics[f"relative_volume_error_{region}"] = relative
        metrics[f"target_region_present_{region}"] = float(target_present)
        metrics[f"predicted_region_present_{region}"] = float(predicted_present)
        metrics[f"false_positive_volume_ml_{region}"] = (
            predicted[region] if not target_present else 0.0
        )
    minimum_voxels = max(1, round(minimum_component_mm3 / float(np.prod(spacing))))
    for prefix, mask in (("predicted", prediction > 0), ("target", target > 0)):
        labeled, count = connected_components(mask)
        metrics[f"{prefix}_component_count_WT"] = float(
            sum((labeled == index).sum() >= minimum_voxels for index in range(1, count + 1))
        )
    metrics["component_count_absolute_error_WT"] = abs(
        metrics["predicted_component_count_WT"] - metrics["target_component_count_WT"]
    )
    predicted_laterality = laterality_from_canonical_mask(prediction)
    target_laterality = laterality_from_canonical_mask(target)
    metrics["predicted_laterality"] = predicted_laterality
    metrics["target_laterality"] = target_laterality
    metrics["laterality_correct"] = float(predicted_laterality == target_laterality)
    return metrics


def aggregate_volume_metrics(
    cases: list[dict], regions: tuple[str, ...] = ("WT", "TC", "ET", "SNFH")
) -> dict[str, float]:
    result = {}
    for region in regions:
        predicted = np.asarray([case[f"predicted_volume_ml_{region}"] for case in cases])
        target = np.asarray([case[f"target_volume_ml_{region}"] for case in cases])
        target_present = target > 0
        target_absent = ~target_present
        result[f"median_absolute_volume_error_ml_{region}"] = float(
            np.median(np.abs(predicted - target))
        )
        result[f"median_relative_volume_error_{region}"] = (
            float(
                np.median(
                    np.abs(predicted[target_present] - target[target_present])
                    / target[target_present]
                )
            )
            if target_present.any()
            else float("nan")
        )
        result[f"ccc_{region}"] = concordance_correlation_coefficient(predicted, target)
        result[f"target_present_cases_{region}"] = float(target_present.sum())
        result[f"target_absent_cases_{region}"] = float(target_absent.sum())
        result[f"region_presence_sensitivity_{region}"] = (
            float(np.mean(predicted[target_present] > 0)) if target_present.any() else float("nan")
        )
        result[f"absent_region_false_positive_rate_{region}"] = (
            float(np.mean(predicted[target_absent] > 0)) if target_absent.any() else float("nan")
        )
        result[f"median_false_positive_volume_ml_{region}"] = (
            float(np.median(predicted[target_absent])) if target_absent.any() else float("nan")
        )
    target_laterality = np.asarray([case["target_laterality"] for case in cases])
    predicted_laterality = np.asarray([case["predicted_laterality"] for case in cases])
    result["laterality_accuracy"] = float(np.mean(predicted_laterality == target_laterality))
    result["laterality_macro_f1"] = float(
        f1_score(
            target_laterality,
            predicted_laterality,
            labels=("left", "right", "bilateral", "midline", "none"),
            average="macro",
            zero_division=0,
        )
    )
    non_ambiguous = np.isin(target_laterality, ("left", "right"))
    result["laterality_nonambiguous_cases"] = float(non_ambiguous.sum())
    result["laterality_nonambiguous_f1"] = (
        float(
            f1_score(
                target_laterality[non_ambiguous],
                predicted_laterality[non_ambiguous],
                labels=("left", "right"),
                average="macro",
                zero_division=0,
            )
        )
        if non_ambiguous.any()
        else float("nan")
    )
    return result


def aggregate_auxiliary_burden_metrics(
    cases: list[dict], regions: tuple[str, ...] = ("WT", "TC", "ET", "SNFH")
) -> dict[str, float]:
    result = {}
    for region in regions:
        predicted = np.asarray(
            [case[f"burden_head_predicted_volume_ml_{region}"] for case in cases]
        )
        target = np.asarray([case[f"target_volume_ml_{region}"] for case in cases])
        present = target > 0
        result[f"burden_head_median_absolute_volume_error_ml_{region}"] = float(
            np.median(np.abs(predicted - target))
        )
        result[f"burden_head_median_relative_volume_error_{region}"] = (
            float(np.median(np.abs(predicted[present] - target[present]) / target[present]))
            if present.any()
            else float("nan")
        )
        result[f"burden_head_ccc_{region}"] = concordance_correlation_coefficient(predicted, target)
    return result
