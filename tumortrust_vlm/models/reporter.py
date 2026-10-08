from __future__ import annotations

import torch
from torch import nn


class RegionAwarePooler(nn.Module):
    def forward(self, feature_map: torch.Tensor, soft_regions: torch.Tensor) -> torch.Tensor:
        regions = torch.nn.functional.interpolate(
            soft_regions, size=feature_map.shape[2:], mode="trilinear", align_corners=False
        )
        denominator = regions.flatten(2).sum(-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bcdhw,brdhw->brc", feature_map, regions) / denominator


class EvidenceConditionedReporter(nn.Module):
    """Compact decoder used before any optional 7B comparison.

    Visual and structured evidence both initialize generation; their gates are exposed so input
    interventions can verify that either source materially affects the output.
    """

    def __init__(
        self,
        vocab_size: int,
        visual_dim: int,
        evidence_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.visual_adapter = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.evidence_adapter = nn.Sequential(nn.Linear(evidence_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.visual_gate = nn.Parameter(torch.tensor(0.0))
        self.evidence_gate = nn.Parameter(torch.tensor(0.0))
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.field_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, evidence_dim)
        )

    def source_contexts(
        self,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
        batch: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        visual_context = (
            self.visual_adapter(visual_tokens.mean(dim=1))
            if visual_tokens is not None
            else None
        )
        evidence_context = self.evidence_adapter(evidence) if evidence is not None else None
        return visual_context, evidence_context

    def context(self, visual_tokens: torch.Tensor | None, evidence: torch.Tensor | None, batch: int) -> torch.Tensor:
        device = self.visual_gate.device
        hidden_dim = self.output.in_features
        result = torch.zeros(batch, hidden_dim, device=device)
        visual_context, evidence_context = self.source_contexts(
            visual_tokens, evidence, batch
        )
        if visual_context is not None:
            result = result + torch.sigmoid(self.visual_gate) * visual_context
        if evidence_context is not None:
            result = result + torch.sigmoid(self.evidence_gate) * evidence_context
        return result

    def reconstruct_fields(
        self,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
        batch: int,
    ) -> torch.Tensor:
        return self.field_head(self.context(visual_tokens, evidence, batch))

    def source_consistency_loss(
        self, visual_tokens: torch.Tensor, evidence: torch.Tensor
    ) -> torch.Tensor:
        visual_context, evidence_context = self.source_contexts(
            visual_tokens, evidence, visual_tokens.shape[0]
        )
        assert visual_context is not None and evidence_context is not None
        return 1 - torch.nn.functional.cosine_similarity(
            visual_context, evidence_context, dim=1
        ).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        visual_tokens: torch.Tensor | None = None,
        evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.context(visual_tokens, evidence, input_ids.shape[0]).unsqueeze(0)
        decoded, _ = self.decoder(self.embedding(input_ids), hidden)
        return self.output(decoded)

    @torch.no_grad()
    def generate(
        self,
        bos_id: int,
        eos_id: int,
        max_length: int,
        *,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if visual_tokens is not None:
            batch = visual_tokens.shape[0]
        elif evidence is not None:
            batch = evidence.shape[0]
        elif batch_size is not None:
            batch = batch_size
        else:
            raise ValueError("batch_size is required when generating without conditioning")
        ids = torch.full((batch, 1), bos_id, dtype=torch.long, device=self.visual_gate.device)
        hidden = self.context(visual_tokens, evidence, batch).unsqueeze(0)
        finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
        for _ in range(max_length - 1):
            output, hidden = self.decoder(self.embedding(ids[:, -1:]), hidden)
            next_id = self.output(output[:, -1]).argmax(-1)
            next_id = torch.where(finished, torch.full_like(next_id, 0), next_id)
            ids = torch.cat((ids, next_id[:, None]), dim=1)
            finished |= next_id == eos_id
            if finished.all():
                break
        return ids
