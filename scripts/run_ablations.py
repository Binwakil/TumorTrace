#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.ablation import materialize_ablation_configs, run_ablation_configs
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--matrix", default="configs/ablations.yaml")
    parser.add_argument("--output", default="artifacts/private/ablation_configs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--enable-trigger",
        action="append",
        default=[],
        help="Enable one named conditional trigger after its scientific criterion is met.",
    )
    args = parser.parse_args()
    paths = materialize_ablation_configs(args.base, args.matrix, args.output)
    results = run_ablation_configs(
        paths,
        args.python,
        dry_run=args.dry_run,
        checkpoint=args.checkpoint,
        enabled_triggers=set(args.enable_trigger),
    )
    atomic_json_dump(results, ROOT / "artifacts" / "ablation_run_status.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
