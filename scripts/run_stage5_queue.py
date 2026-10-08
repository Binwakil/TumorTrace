#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.experiment_queue import run_queue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or exactly resume the dependency-approved stage-5 experiment queue."
    )
    parser.add_argument(
        "--registry", default="artifacts/private/stage5_configs/registry.json"
    )
    parser.add_argument(
        "--status", default="artifacts/private/stage5_queue_status.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    result = run_queue(
        Path(args.registry), Path(args.status), device=args.device, python=args.python
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
