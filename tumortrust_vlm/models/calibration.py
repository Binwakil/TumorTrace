from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        logits_tensor = torch.as_tensor(logits, dtype=torch.float64)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)

        def objective(log_temperature: float) -> float:
            temperature = np.exp(log_temperature)
            return float(torch.nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor))

        result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) / self.temperature

    def state_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}


class ConformalResidualInterval:
    def __init__(self, coverage: float = 0.90) -> None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        self.coverage = coverage
        self.quantile: float | None = None

    def fit(self, predicted: np.ndarray, target: np.ndarray) -> ConformalResidualInterval:
        residuals = np.abs(np.asarray(target) - np.asarray(predicted))
        count = len(residuals)
        level = min(1.0, np.ceil((count + 1) * self.coverage) / count)
        self.quantile = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict(self, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile is None:
            raise RuntimeError("Interval must be fit before prediction")
        predicted = np.asarray(predicted)
        return np.maximum(0, predicted - self.quantile), predicted + self.quantile

