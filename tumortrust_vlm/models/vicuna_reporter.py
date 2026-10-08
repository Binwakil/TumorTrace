from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
)


def local_decoder_family(model_path: str | Path) -> str:
    """Resolve the language backbone without importing either LLaVA codebase."""
    payload = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    model_type = payload.get("model_type")
    if model_type == "llava":
        return "llama"
    if model_type == "llava_mistral":
        return "mistral"
    raise ValueError(f"Unsupported local LLaVA model_type: {model_type!r}")


class PrefixConditionedCausalReporter(nn.Module):
    """LoRA language-decoder comparator with new 3D/evidence prefix adapters.

    This deliberately loads only the language weights. It does not instantiate CLIP and does not
    reuse the inherited LLaVA projector that failed in NeuroGround.
    """

    def __init__(
        self,
        model_path: str,
        visual_dim: int,
        evidence_dim: int,
        prefix_tokens: int = 8,
        context_dim: int = 512,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        dtype: torch.dtype = torch.float16,
        decoder_family: str | None = None,
    ) -> None:
        super().__init__()
        model_payload = json.loads(
            (Path(model_path) / "config.json").read_text(encoding="utf-8")
        )
        resolved_family = decoder_family or local_decoder_family(model_path)
        if resolved_family == "llama":
            config = LlamaConfig.from_dict(model_payload)
            model_class = LlamaForCausalLM
        elif resolved_family == "mistral":
            config = MistralConfig.from_dict(model_payload)
            model_class = MistralForCausalLM
        else:
            raise ValueError(f"Unsupported decoder family: {resolved_family!r}")
        language_model = model_class.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        language_model.config.use_cache = False
        for parameter in language_model.parameters():
            parameter.requires_grad = False
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        )
        self.language_model = get_peft_model(language_model, lora)
        self.prefix_tokens = prefix_tokens
        self.context_dim = context_dim
        self.hidden_size = config.hidden_size
        self.decoder_family = resolved_family
        self.visual_adapter = nn.Sequential(
            nn.Linear(visual_dim, context_dim), nn.GELU(), nn.LayerNorm(context_dim)
        )
        self.evidence_adapter = nn.Sequential(
            nn.Linear(evidence_dim, context_dim), nn.GELU(), nn.LayerNorm(context_dim)
        )
        self.prefix_projection = nn.Linear(context_dim, prefix_tokens * config.hidden_size)

    def context(
        self,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        context = torch.zeros(batch_size, self.context_dim, device=device)
        if visual_tokens is not None:
            context = context + self.visual_adapter(visual_tokens.mean(dim=1).float())
        if evidence is not None:
            context = context + self.evidence_adapter(evidence.float())
        return context

    def prefix_embeddings(
        self,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        context = self.context(visual_tokens, evidence, batch_size, device)
        return (
            self.prefix_projection(context)
            .view(batch_size, self.prefix_tokens, self.hidden_size)
            .to(dtype)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
    ):
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        prefix = self.prefix_embeddings(
            visual_tokens,
            evidence,
            input_ids.shape[0],
            input_ids.device,
            embeddings.dtype,
        )
        inputs_embeds = torch.cat((prefix, embeddings), dim=1)
        prefix_attention = torch.ones(
            input_ids.shape[0],
            self.prefix_tokens,
            dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        combined_attention = torch.cat((prefix_attention, attention_mask), dim=1)
        prefix_labels = torch.full(
            (input_ids.shape[0], self.prefix_tokens), -100, dtype=labels.dtype, device=labels.device
        )
        combined_labels = torch.cat((prefix_labels, labels), dim=1)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        visual_tokens: torch.Tensor | None,
        evidence: torch.Tensor | None,
        maximum_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Greedy generation from the learned 3D/evidence prefix."""
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        prefix = self.prefix_embeddings(
            visual_tokens,
            evidence,
            input_ids.shape[0],
            input_ids.device,
            embeddings.dtype,
        )
        combined_attention = torch.cat(
            (
                torch.ones(
                    input_ids.shape[0],
                    self.prefix_tokens,
                    dtype=attention_mask.dtype,
                    device=input_ids.device,
                ),
                attention_mask,
            ),
            dim=1,
        )
        output = self.language_model(
            inputs_embeds=torch.cat((prefix, embeddings), dim=1),
            attention_mask=combined_attention,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1)
        generated = [next_token]
        finished = next_token == eos_token_id
        for _ in range(maximum_new_tokens - 1):
            if finished.all():
                break
            combined_attention = torch.cat(
                (
                    combined_attention,
                    torch.ones(
                        input_ids.shape[0],
                        1,
                        dtype=combined_attention.dtype,
                        device=input_ids.device,
                    ),
                ),
                dim=1,
            )
            output = self.language_model(
                input_ids=next_token[:, None],
                attention_mask=combined_attention,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_token = output.logits[:, -1].argmax(-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_token_id), next_token
            )
            generated.append(next_token)
            finished |= next_token == eos_token_id
        return torch.stack(generated, dim=1)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        known = dict(self.named_parameters())
        unexpected = sorted(set(state) - set(known))
        if unexpected:
            raise ValueError(f"Unknown trainable checkpoint keys: {unexpected[:5]}")
        for name, value in state.items():
            known[name].data.copy_(value.to(device=known[name].device, dtype=known[name].dtype))

    def trainable_parameter_summary(self) -> dict[str, int]:
        return {
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }


# Backward-compatible name retained for existing checkpoints and scripts. Public tables and new
# artifacts use descriptive model names rather than this historical implementation label.
VicunaEvidenceReporter = PrefixConditionedCausalReporter
