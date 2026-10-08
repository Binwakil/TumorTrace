#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.nnunet_finalize import finalize_nnunet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preserve, evaluate, and compare final/best nnU-Net validation predictions."
    )
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    result = finalize_nnunet(
        ROOT,
        python=args.python,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
