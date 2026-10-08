#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tumortrust_vlm.config import load_config
from tumortrust_vlm.models.core import MultiTaskSegResNet
from tumortrust_vlm.models.losses import MultiTaskObjective
from tumortrust_vlm.models.reporter import EvidenceConditionedReporter, RegionAwarePooler
from tumortrust_vlm.reporting.evidence import build_evidence_card
from tumortrust_vlm.reporting.renderer import render_findings, validate_rendered_values
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    config = load_config("configs/smoke.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskSegResNet(init_filters=8).to(device)
    objective = MultiTaskObjective("fixed", 0.2).to(device)
    image = torch.randn(1, 4, 32, 32, 32, device=device)
    label = torch.randint(0, 4, (1, 32, 32, 32), device=device)
    class_label = torch.tensor([1], device=device)
    output = model(image)
    losses = objective(output, label, class_label)
    losses["loss"].backward()
    gradients = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if not any(name.startswith("backbone") and value > 0 for name, value in gradients.items()):
        raise AssertionError("No backbone gradient")
    if not any(name.startswith("classifier") and value > 0 for name, value in gradients.items()):
        raise AssertionError("No classifier gradient")

    pooler = RegionAwarePooler().to(device)
    # Reporter training consumes cached/frozen core features; detach to model that stage boundary.
    region_tokens = pooler(output["latent"].detach(), output["segmentation"].softmax(1).detach())
    reporter = EvidenceConditionedReporter(32, region_tokens.shape[-1], 12, hidden_dim=64, embedding_dim=32).to(device)
    ids = torch.randint(0, 32, (1, 8), device=device)
    report_logits = reporter(ids, visual_tokens=region_tokens, evidence=torch.randn(1, 12, device=device))
    report_logits.sum().backward()

    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[4:10, 10:18, 12:20] = 3
    card = build_evidence_card(
        "synthetic",
        mask,
        (1, 1, 1),
        {"GLI": 0.7, "MEN": 0.2, "MET": 0.1},
        segmentation_uncertainty=0.2,
        classification_uncertainty=0.1,
        referral_threshold=0.5,
    )
    rendered = render_findings(card)
    validate_rendered_values(card, rendered)
    result = {
        "device": str(device),
        "segmentation_shape": list(output["segmentation"].shape),
        "classification_shape": list(output["classification"].shape),
        "region_token_shape": list(region_tokens.shape),
        "report_shape": list(report_logits.shape),
        "trainable_modules_with_gradients": len(gradients),
        "renderer_trace_passed": True,
        "config_loaded": config["project"]["name"],
    }
    atomic_json_dump(result, ROOT / "artifacts" / "smoke_test.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
