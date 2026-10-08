from __future__ import annotations

import torch
from monai.networks.nets import SegResNet
from torch import nn


class MultiTaskSegResNet(nn.Module):
    """One 3D encoder with parallel segmentation and tumor-family heads.

    The classification output never conditions the segmentation path. Modality-presence flags are
    appended only to the classification representation so missing-sequence state is explicit.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        classification_classes: int = 3,
        init_filters: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
        latent_channels = init_filters * 8
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_channels + in_channels),
            nn.Linear(latent_channels + in_channels, latent_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_channels, classification_classes),
        )

    def classify_latent(
        self,
        latent: torch.Tensor,
        modality_presence: torch.Tensor | None,
        input_channels: int = 4,
    ) -> torch.Tensor:
        pooled = latent.mean(dim=(2, 3, 4))
        if modality_presence is None:
            modality_presence = torch.ones(
                latent.shape[0], input_channels, device=latent.device, dtype=pooled.dtype
            )
        return self.classifier(torch.cat((pooled, modality_presence.to(pooled.dtype)), dim=1))

    def encode_for_classification(self, image: torch.Tensor) -> torch.Tensor:
        latent, _ = self.backbone.encode(image)
        return latent

    def encode_for_reporting_and_classification(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode once when classification and reporting share the image encoder."""
        latent = self.encode_for_classification(image)
        return latent, latent

    def segment(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, skips = self.backbone.encode(image)
        return self.backbone.decode(latent, list(reversed(skips))), latent

    def forward(
        self, image: torch.Tensor, modality_presence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        segmentation, latent = self.segment(image)
        classification = self.classify_latent(latent, modality_presence, image.shape[1])
        return {
            "segmentation": segmentation,
            "classification": classification,
            "latent": latent,
        }


class SeparateEncoderMultiTaskSegResNet(MultiTaskSegResNet):
    """Negative-transfer control with a distinct classification encoder."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        classification_classes: int = 3,
        init_filters: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            classification_classes,
            init_filters,
            dropout,
        )
        self.classification_backbone = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )

    def encode_for_classification(self, image: torch.Tensor) -> torch.Tensor:
        latent, _ = self.classification_backbone.encode(image)
        return latent

    def encode_for_reporting_and_classification(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reporting_latent, _ = self.backbone.encode(image)
        classification_latent = self.encode_for_classification(image)
        return reporting_latent, classification_latent

    def forward(
        self, image: torch.Tensor, modality_presence: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        segmentation, segmentation_latent = self.segment(image)
        classification_latent = self.encode_for_classification(image)
        classification = self.classify_latent(
            classification_latent, modality_presence, image.shape[1]
        )
        return {
            "segmentation": segmentation,
            "classification": classification,
            "latent": segmentation_latent,
            "classification_latent": classification_latent,
        }


def assert_parallel_heads(
    model: MultiTaskSegResNet, shape: tuple[int, ...] = (1, 4, 32, 32, 32)
) -> None:
    image = torch.randn(shape, requires_grad=True)
    output = model(image)
    classification_gradient = torch.autograd.grad(
        output["classification"].sum(), output["segmentation"], allow_unused=True
    )[0]
    if classification_gradient is not None:
        raise AssertionError("Classification output conditions the segmentation output")
