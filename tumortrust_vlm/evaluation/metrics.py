from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as connected_components
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from tumortrust_vlm.data.constants import REGION_LABELS


def region_mask(segmentation: np.ndarray, region: str) -> np.ndarray:
    return np.isin(segmentation, REGION_LABELS[region])


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def hd95(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")
    pred_to_target, target_to_pred = _surface_distances(prediction, target, spacing)
    distances = np.concatenate((pred_to_target, target_to_pred))
    return float(np.percentile(distances, 95))


def surface_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    pred_to_target, target_to_pred = _surface_distances(prediction, target, spacing)
    numerator = (pred_to_target <= tolerance_mm).sum() + (target_to_pred <= tolerance_mm).sum()
    return float(numerator / (len(pred_to_target) + len(target_to_pred)))


def _surface_distances(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    pred_surface = np.logical_xor(prediction, binary_erosion(prediction))
    target_surface = np.logical_xor(target, binary_erosion(target))
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    pred_to_target = distance_to_target[pred_surface]
    del distance_to_target
    distance_to_prediction = distance_transform_edt(~pred_surface, sampling=spacing)
    target_to_pred = distance_to_prediction[target_surface]
    return pred_to_target, target_to_pred


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0, 1.0
    if not prediction.any() or not target.any():
        return float("inf"), 0.0
    pred_to_target, target_to_pred = _surface_distances(prediction, target, spacing)
    distances = np.concatenate((pred_to_target, target_to_pred))
    numerator = (pred_to_target <= tolerance_mm).sum() + (target_to_pred <= tolerance_mm).sum()
    return float(np.percentile(distances, 95)), float(numerator / len(distances))


def _components(mask: np.ndarray, minimum_voxels: int) -> list[np.ndarray]:
    labeled, count = connected_components(mask)
    return [
        (labeled == index)
        for index in range(1, count + 1)
        if (labeled == index).sum() >= minimum_voxels
    ]


def lesion_detection(
    prediction: np.ndarray,
    target: np.ndarray,
    minimum_voxels: int = 100,
    small_lesion_maximum_voxels: int | None = None,
) -> dict[str, float]:
    predicted = _components(prediction, minimum_voxels)
    targets = _components(target, minimum_voxels)
    matched_pred: set[int] = set()
    matched_target: set[int] = set()
    candidates = []
    for pred_index, pred in enumerate(predicted):
        for target_index, truth in enumerate(targets):
            intersection = np.logical_and(pred, truth).sum()
            union = np.logical_or(pred, truth).sum()
            if intersection:
                candidates.append((intersection / union, pred_index, target_index))
    for _, pred_index, target_index in sorted(candidates, reverse=True):
        if pred_index not in matched_pred and target_index not in matched_target:
            matched_pred.add(pred_index)
            matched_target.add(target_index)
    tp = len(matched_target)
    fp = len(predicted) - tp
    fn = len(targets) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "lesion_precision": precision,
        "lesion_recall": recall,
        "lesion_f1": f1,
        "lesion_tp": tp,
        "lesion_fp": fp,
        "lesion_fn": fn,
    }
    if small_lesion_maximum_voxels is not None:
        if small_lesion_maximum_voxels <= minimum_voxels:
            raise ValueError("Small-lesion maximum must exceed the minimum component size")
        small_targets = {
            index
            for index, component in enumerate(targets)
            if component.sum() < small_lesion_maximum_voxels
        }
        large_targets = set(range(len(targets))) - small_targets
        small_tp = len(small_targets & matched_target)
        large_tp = len(large_targets & matched_target)
        result.update(
            {
                "small_lesion_targets": len(small_targets),
                "small_lesion_tp": small_tp,
                "small_lesion_recall": (
                    small_tp / len(small_targets) if small_targets else float("nan")
                ),
                "large_lesion_targets": len(large_targets),
                "large_lesion_tp": large_tp,
                "large_lesion_recall": (
                    large_tp / len(large_targets) if large_targets else float("nan")
                ),
            }
        )
    return result


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float] = (1, 1, 1),
    regions: tuple[str, ...] = ("WT", "TC", "ET", "SNFH"),
    minimum_lesion_mm3: float = 100,
    small_lesion_maximum_mm3: float = 1000,
) -> dict[str, float]:
    result: dict[str, float] = {}
    voxel_volume = float(np.prod(spacing))
    minimum_voxels = max(1, round(minimum_lesion_mm3 / voxel_volume))
    for region in regions:
        pred_region = region_mask(prediction, region)
        target_region = region_mask(target, region)
        result[f"dice_{region}"] = dice_score(pred_region, target_region)
        region_hd95, region_surface_dice = surface_metrics(pred_region, target_region, spacing)
        result[f"hd95_{region}"] = region_hd95
        result[f"surface_dice_{region}"] = region_surface_dice
        if region == "WT":
            result.update(
                lesion_detection(
                    pred_region,
                    target_region,
                    minimum_voxels,
                    max(1, round(small_lesion_maximum_mm3 / voxel_volume)),
                )
            )
    result["macro_dice"] = float(np.mean([result[f"dice_{region}"] for region in regions]))
    return result


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    boundaries = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in pairwise(boundaries):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, ece_bins: int = 15
) -> dict[str, float | list | dict]:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    predicted = probabilities.argmax(axis=1)
    onehot = np.eye(probabilities.shape[1])[labels]
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "ece": expected_calibration_error(probabilities, labels, bins=ece_bins),
        "brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "nll": float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1)))
        ),
    }
    matrix = confusion_matrix(labels, predicted, labels=np.arange(probabilities.shape[1]))
    result["confusion_matrix"] = matrix.tolist()
    classwise = {}
    for class_index in range(probabilities.shape[1]):
        tp = int(matrix[class_index, class_index])
        fn = int(matrix[class_index].sum() - tp)
        fp = int(matrix[:, class_index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        binary_target = labels == class_index
        payload = {
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        }
        if len(np.unique(binary_target)) == 2:
            payload["auroc"] = float(roc_auc_score(binary_target, probabilities[:, class_index]))
            payload["auprc"] = float(
                average_precision_score(binary_target, probabilities[:, class_index])
            )
        classwise[str(class_index)] = payload
    result["classwise"] = classwise
    try:
        result["macro_auroc"] = float(
            roc_auc_score(onehot, probabilities, average="macro", multi_class="ovr")
        )
        result["macro_auprc"] = float(
            average_precision_score(onehot, probabilities, average="macro")
        )
    except ValueError:
        result["macro_auroc"] = float("nan")
        result["macro_auprc"] = float("nan")
    return result


def aggregate_case_metrics(cases: list[dict]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for key, value in case.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                values[key].append(float(value))
    return {key: float(np.mean(items)) for key, items in values.items() if items}
