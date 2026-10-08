import numpy as np

from tumortrust_vlm.evaluation.metrics import (
    classification_metrics,
    hd95,
    lesion_detection,
    segmentation_metrics,
    surface_dice,
)
from tumortrust_vlm.evaluation.statistics import holm_adjust, paired_bootstrap_difference
from tumortrust_vlm.evaluation.trust import (
    qu_brats_score,
    random_referral_control,
    risk_coverage,
    selective_risk_at_coverage,
)


def test_segmentation_identity_is_perfect():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 3
    metrics = segmentation_metrics(mask, mask)
    assert metrics["macro_dice"] == 1.0
    assert metrics["hd95_WT"] == 0.0
    assert metrics["lesion_f1"] == 1.0


def test_classification_and_statistics():
    probabilities = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    labels = np.array([0, 1, 2])
    metrics = classification_metrics(probabilities, labels, ece_bins=3)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["classwise"]["0"]["specificity"] == 1.0
    comparison = paired_bootstrap_difference(np.array([0.9, 0.8]), np.array([0.8, 0.7]), samples=50)
    assert comparison["mean_difference"] > 0
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert all(0 <= value <= 1 for value in adjusted)


def test_selective_risk_orders_low_uncertainty_first():
    uncertainty = np.array([0.1, 0.2, 0.9])
    error = np.array([0.0, 0.1, 1.0])
    curve = risk_coverage(uncertainty, error)
    assert curve["risk"][0] == 0
    assert selective_risk_at_coverage(uncertainty, error, 2 / 3) < error.mean()
    control = random_referral_control(uncertainty, error, 2 / 3, samples=100, seed=1)
    assert control["selective_risk"] < control["random_referral_mean_risk"]


def test_surface_distances_respect_voxel_spacing():
    target = np.zeros((8, 8, 8), dtype=bool)
    prediction = np.zeros_like(target)
    target[2, 3, 3] = True
    prediction[3, 3, 3] = True

    assert hd95(prediction, target, (2.0, 1.0, 1.0)) == 2.0
    assert surface_dice(prediction, target, (2.0, 1.0, 1.0), tolerance_mm=1.0) == 0.0


def test_qu_brats_perfect_certain_prediction_scores_one():
    target = np.zeros((8, 8, 8), dtype=bool)
    target[2:5, 2:5, 2:5] = True
    result = qu_brats_score(
        target, target, np.zeros_like(target, dtype=float), np.ones_like(target)
    )

    assert result["score"] == 1.0


def test_small_lesion_recall_is_reported_separately() -> None:
    target = np.zeros((30, 30, 30), dtype=bool)
    target[1:4, 1:4, 1:4] = True
    target[10:20, 10:20, 10:20] = True
    prediction = np.zeros_like(target)
    prediction[10:20, 10:20, 10:20] = True

    result = lesion_detection(
        prediction,
        target,
        minimum_voxels=10,
        small_lesion_maximum_voxels=100,
    )

    assert result["small_lesion_targets"] == 1
    assert result["small_lesion_recall"] == 0.0
    assert result["large_lesion_targets"] == 1
    assert result["large_lesion_recall"] == 1.0
