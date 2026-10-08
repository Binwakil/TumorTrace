#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.evaluation.radiologist import summarize_radiologist_scores
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--private-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    private_map = json.loads(Path(args.private_map).read_text(encoding="utf-8"))
    result = summarize_radiologist_scores(
        scores, private_map, bootstrap_samples=args.bootstrap_samples
    )
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
