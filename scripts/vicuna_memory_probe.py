#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from tumortrust_vlm.models.vicuna_reporter import (
    PrefixConditionedCausalReporter,
    local_decoder_family,
)
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../shared/models/llava-v1.5-7b")
    parser.add_argument("--output", default="artifacts/vicuna_memory_probe.json")
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--visual-dim", type=int, default=128)
    parser.add_argument("--evidence-dim", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    decoder_family = local_decoder_family(args.model)
    reporter = PrefixConditionedCausalReporter(
        args.model,
        args.visual_dim,
        args.evidence_dim,
        decoder_family=decoder_family,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in reporter.parameters() if parameter.requires_grad], lr=2e-4
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    input_ids = torch.full((1, args.sequence_length), pad_id, dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    labels = input_ids.clone()
    visual = torch.randn(1, 4, args.visual_dim, device=device, dtype=torch.float16)
    evidence = torch.randn(1, args.evidence_dim, device=device)
    output = reporter(input_ids, attention, labels, visual, evidence)
    output.loss.backward()
    optimizer.step()
    reporter.eval()
    prompt_ids = input_ids[:, :8]
    prompt_attention = torch.ones_like(prompt_ids)
    generated = reporter.generate(
        prompt_ids,
        prompt_attention,
        visual_tokens=visual,
        evidence=evidence,
        maximum_new_tokens=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
    )
    trainable_state = reporter.trainable_state_dict()
    torch.cuda.synchronize()
    result = {
        **reporter.trainable_parameter_summary(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "sequence_length": args.sequence_length,
        "generation_shape": list(generated.shape),
        "trainable_checkpoint_mib": sum(
            tensor.numel() * tensor.element_size() for tensor in trainable_state.values()
        )
        / 2**20,
        "elapsed_seconds": time.perf_counter() - start,
        "optimizer_inclusive": True,
        "clip_loaded": False,
        "inherited_llava_projector_loaded": False,
        "decoder_family": decoder_family,
        "model_path": args.model,
        "passed": True,
    }
    atomic_json_dump(result, ROOT / args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
