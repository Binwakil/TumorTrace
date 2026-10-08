from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from tumortrust_vlm.evaluation.metrics import dice_score


def entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities), 1e-8, 1)
    return -np.sum(probabilities * np.log(probabilities), axis=axis)


def failure_detection(uncertainty: np.ndarray, error: np.ndarray, threshold: float) -> dict[str, float]:
    failures = np.asarray(error) >= threshold
    uncertainty = np.asarray(uncertainty)
    if len(np.unique(failures)) < 2:
        return {"failure_auroc": float("nan"), "failure_auprc": float("nan")}
    return {
        "failure_auroc": float(roc_auc_score(failures, uncertainty)),
        "failure_auprc": float(average_precision_score(failures, uncertainty)),
    }


def risk_coverage(uncertainty: np.ndarray, error: np.ndarray) -> dict[str, np.ndarray]:
    uncertainty = np.asarray(uncertainty)
    error = np.asarray(error)
    order = np.argsort(uncertainty)
    cumulative = np.cumsum(error[order])
    counts = np.arange(1, len(error) + 1)
    return {"coverage": counts / len(error), "risk": cumulative / counts}


def selective_risk_at_coverage(uncertainty: np.ndarray, error: np.ndarray, coverage: float = 0.8) -> float:
    curve = risk_coverage(uncertainty, error)
    index = max(0, int(np.ceil(len(error) * coverage)) - 1)
    return float(curve["risk"][index])


def random_referral_control(
    uncertainty: np.ndarray,
    error: np.ndarray,
    coverage: float = 0.8,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    error = np.asarray(error, dtype=float)
    retained = max(1, int(np.ceil(len(error) * coverage)))
    selective = selective_risk_at_coverage(uncertainty, error, coverage)
    rng = np.random.default_rng(seed)
    random_risks = np.asarray(
        [rng.choice(error, size=retained, replace=False).mean() for _ in range(samples)]
    )
    baseline = float(error.mean())
    return {
        "unreferred_risk": baseline,
        "selective_risk": selective,
        "selective_relative_risk_reduction": (
            float((baseline - selective) / baseline) if baseline else 0.0
        ),
        "random_referral_mean_risk": float(random_risks.mean()),
        "random_referral_p_value": float(
            (np.sum(random_risks <= selective) + 1) / (samples + 1)
        ),
    }


def ensemble_uncertainty(probability_samples: np.ndarray) -> dict[str, np.ndarray]:
    samples = np.asarray(probability_samples)
    mean_probability = samples.mean(axis=0)
    predictive_entropy = entropy(mean_probability, axis=1)
    expected_entropy = entropy(samples, axis=2).mean(axis=0)
    return {
        "mean_probability": mean_probability,
        "predictive_entropy": predictive_entropy,
        "mutual_information": predictive_entropy - expected_entropy,
    }


def qu_brats_score(
    prediction: np.ndarray,
    target: np.ndarray,
    uncertainty: np.ndarray,
    brain_mask: np.ndarray,
    points: int = 40,
) -> dict[str, float | list[float]]:
    """Compute the official QU-BraTS Dice/FTP/FTN threshold-AUC score.

    This follows the challenge reference implementation with uncertainty normalized to [0, 1].
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    uncertainty = np.clip(np.asarray(uncertainty, dtype=float), 0.0, 1.0)
    brain_mask = np.asarray(brain_mask, dtype=bool)
    thresholds = np.linspace(1.0, 0.0, points + 1)
    true_positive = prediction & target & brain_mask
    true_negative = ~prediction & ~target & brain_mask
    tp_total = max(1, int(true_positive.sum()))
    tn_total = max(1, int(true_negative.sum()))
    dice_values = []
    ftp_values = []
    ftn_values = []
    for threshold in thresholds:
        certain = uncertainty <= threshold
        dice_values.append(dice_score(prediction & certain, target & certain))
        ftp_values.append(float((true_positive & ~certain).sum() / tp_total))
        ftn_values.append(float((true_negative & ~certain).sum() / tn_total))
    ascending = thresholds[::-1]
    dice_auc = float(np.trapz(np.asarray(dice_values)[::-1], ascending))
    ftp_auc = float(np.trapz(np.asarray(ftp_values)[::-1], ascending))
    ftn_auc = float(np.trapz(np.asarray(ftn_values)[::-1], ascending))
    score = (dice_auc + (1 - ftp_auc) + (1 - ftn_auc)) / 3
    return {
        "score": float(score),
        "dice_auc": dice_auc,
        "ftp_auc": ftp_auc,
        "ftn_auc": ftn_auc,
        "thresholds": thresholds.tolist(),
        "dice": dice_values,
        "ftp": ftp_values,
        "ftn": ftn_values,
    }
