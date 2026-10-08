from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, samples: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap_ci expects a non-empty 1D array")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    alpha = 1 - confidence
    return tuple(float(value) for value in np.quantile(estimates, (alpha / 2, 1 - alpha / 2)))


def paired_bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("Paired arrays must have identical shapes")
    differences = candidate - baseline
    low, high = bootstrap_ci(differences, samples=samples, confidence=confidence, seed=seed)
    return {"mean_difference": float(differences.mean()), "ci_low": low, "ci_high": high}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()

