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

from tumortrust_vlm.models.core import MultiTaskSegResNet
from tumortrust_vlm.models.losses import MultiTaskObjective
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--filters", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the memory gate")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = MultiTaskSegResNet(init_filters=args.filters).to(device)
    objective = MultiTaskObjective().to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(objective.parameters()), lr=2e-4)
    image = torch.randn(1, 4, args.patch, args.patch, args.patch, device=device)
    label = torch.randint(0, 4, (1, args.patch, args.patch, args.patch), device=device)
    target_class = torch.tensor([0], device=device)
    start = time.perf_counter()
    with torch.amp.autocast("cuda"):
        output = model(image)
        loss = objective(output, label, target_class)["loss"]
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "patch": args.patch,
        "init_filters": args.filters,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "optimizer_inclusive": True,
        "elapsed_seconds": time.perf_counter() - start,
        "passed": True,
    }
    atomic_json_dump(result, ROOT / "artifacts" / "memory_probe.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

