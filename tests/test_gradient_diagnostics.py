import math

import torch
from torch import nn

from tumortrust_vlm.models.losses import gradient_diagnostics


class PartiallySharedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 3, bias=False)
        self.segmentation_head = nn.Linear(3, 1, bias=False)
        self.classification_head = nn.Linear(3, 1, bias=False)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.shared(value)
        return self.segmentation_head(latent), self.classification_head(latent)


def test_gradient_diagnostics_aligns_disjoint_heads() -> None:
    model = PartiallySharedModel()
    segmentation, classification = model(torch.ones(2, 3))
    diagnostics = gradient_diagnostics(model, segmentation.sum(), classification.sum())

    assert diagnostics["segmentation_gradient_norm"] > 0
    assert diagnostics["classification_gradient_norm"] > 0
    assert math.isfinite(diagnostics["gradient_cosine"])
    assert -1.0 <= diagnostics["gradient_cosine"] <= 1.0
