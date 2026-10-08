from __future__ import annotations

import torch
from monai.losses import DiceCELoss
from torch import nn
from torch.nn import functional as F


class MultiTaskObjective(nn.Module):
    def __init__(self, mode: str = "fixed", lambda_cls: float = 0.2) -> None:
        super().__init__()
        self.mode = mode
        self.lambda_cls = lambda_cls
        self.segmentation_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.log_variances = nn.Parameter(torch.zeros(2), requires_grad=mode == "uncertainty")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        label: torch.Tensor,
        class_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        classification = F.cross_entropy(outputs["classification"], class_label)
        if self.mode == "classification_only":
            segmentation = classification.detach().new_zeros(())
            total = classification
        else:
            if label.ndim == 4:
                label = label[:, None]
            segmentation = self.segmentation_loss(outputs["segmentation"], label)
        if self.mode == "segmentation_only":
            total = segmentation
        elif self.mode == "uncertainty":
            total = (
                torch.exp(-self.log_variances[0]) * segmentation
                + self.log_variances[0]
                + torch.exp(-self.log_variances[1]) * classification
                + self.log_variances[1]
            )
        elif self.mode in {"fixed", "pcgrad"}:
            total = segmentation + self.lambda_cls * classification
        elif self.mode != "classification_only":
            raise ValueError(f"Unknown multitask loss mode: {self.mode}")
        return {
            "loss": total,
            "segmentation_loss": segmentation,
            "classification_loss": classification,
        }


def gradient_diagnostics(
    model: nn.Module,
    segmentation_loss: torch.Tensor,
    classification_loss: torch.Tensor,
) -> dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    seg_gradients = torch.autograd.grad(
        segmentation_loss, parameters, retain_graph=True, allow_unused=True
    )
    cls_gradients = torch.autograd.grad(
        classification_loss, parameters, retain_graph=True, allow_unused=True
    )
    aligned_segmentation: list[torch.Tensor] = []
    aligned_classification: list[torch.Tensor] = []
    for parameter, seg_gradient, cls_gradient in zip(
        parameters, seg_gradients, cls_gradients, strict=True
    ):
        if seg_gradient is None and cls_gradient is None:
            continue
        aligned_segmentation.append(
            seg_gradient.reshape(-1)
            if seg_gradient is not None
            else torch.zeros_like(parameter).reshape(-1)
        )
        aligned_classification.append(
            cls_gradient.reshape(-1)
            if cls_gradient is not None
            else torch.zeros_like(parameter).reshape(-1)
        )
    if not aligned_segmentation:
        return {
            "segmentation_gradient_norm": float("nan"),
            "classification_gradient_norm": float("nan"),
            "gradient_cosine": float("nan"),
        }
    seg_flat = torch.cat(aligned_segmentation)
    cls_flat = torch.cat(aligned_classification)
    cosine = F.cosine_similarity(seg_flat, cls_flat, dim=0)
    return {
        "segmentation_gradient_norm": float(seg_flat.norm().detach()),
        "classification_gradient_norm": float(cls_flat.norm().detach()),
        "gradient_cosine": float(cosine.detach()),
    }


def pcgrad_backward(
    model: nn.Module,
    segmentation_loss: torch.Tensor,
    classification_loss: torch.Tensor,
    *,
    classification_weight: float = 1.0,
    accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Accumulate two-task PCGrad gradients while preserving task-specific parameters.

    When `scaler` is provided and enabled, both task losses are scaled before their
    per-task `torch.autograd.grad` calls, exactly as a normal `scaler.scale(loss).backward()`
    would, to avoid AMP gradient underflow under autocast. The projection ratios (dot over
    each task's squared norm) are scale-invariant, so this does not change which parameter
    updates get projected; it only protects the raw per-parameter gradients from vanishing in
    fp16. The written `.grad` values remain scaled, so the caller must finish the optimizer
    step with `scaler.step(optimizer)` / `scaler.update()` rather than calling
    `optimizer.step()` directly, exactly like the other objective modes.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    scale_losses = scaler is not None and scaler.is_enabled()
    segmentation_target = scaler.scale(segmentation_loss) if scale_losses else segmentation_loss
    classification_target = (
        scaler.scale(classification_loss * classification_weight)
        if scale_losses
        else classification_loss * classification_weight
    )
    segmentation_gradients = torch.autograd.grad(
        segmentation_target,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    classification_gradients = torch.autograd.grad(
        classification_target,
        parameters,
        allow_unused=True,
    )
    shared = [
        (segmentation, classification)
        for segmentation, classification in zip(
            segmentation_gradients, classification_gradients, strict=True
        )
        if segmentation is not None and classification is not None
    ]
    if shared:
        dot = sum((segmentation * classification).sum() for segmentation, classification in shared)
        segmentation_norm_sq = sum(
            (segmentation * segmentation).sum() for segmentation, _ in shared
        ).clamp_min(1e-12)
        classification_norm_sq = sum(
            (classification * classification).sum() for _, classification in shared
        ).clamp_min(1e-12)
        conflict = bool(dot.detach() < 0)
    else:
        dot = segmentation_loss.new_zeros(())
        segmentation_norm_sq = segmentation_loss.new_zeros(())
        classification_norm_sq = segmentation_loss.new_zeros(())
        conflict = False

    for parameter, segmentation, classification in zip(
        parameters,
        segmentation_gradients,
        classification_gradients,
        strict=True,
    ):
        if segmentation is None and classification is None:
            continue
        if segmentation is None:
            combined = classification
        elif classification is None:
            combined = segmentation
        elif conflict:
            original_segmentation = segmentation
            projected_segmentation = segmentation - dot / classification_norm_sq * classification
            projected_classification = (
                classification - dot / segmentation_norm_sq * original_segmentation
            )
            combined = projected_segmentation + projected_classification
        else:
            combined = segmentation + classification
        assert combined is not None
        contribution = combined.detach() / accumulation_steps
        if parameter.grad is None:
            parameter.grad = contribution.clone()
        else:
            parameter.grad.add_(contribution)
    cosine = dot / (segmentation_norm_sq.sqrt() * classification_norm_sq.sqrt()).clamp_min(1e-12)
    return {
        "pcgrad_conflict": float(conflict),
        "pcgrad_pre_projection_cosine": float(cosine.detach()) if shared else float("nan"),
    }
