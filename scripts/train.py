#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.engine import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    args = parser.parse_args()
    summary = train(load_config(args.config), device=args.device, resume=args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
