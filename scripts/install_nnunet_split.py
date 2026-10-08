#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--preprocessed-root", required=True)
    args = parser.parse_args()
    source = ROOT / "artifacts" / "private" / f"nnunet_{args.dataset_id}_splits_final.json"
    dataset_dirs = list(Path(args.preprocessed_root).glob(f"Dataset{args.dataset_id:03d}_*"))
    if len(dataset_dirs) != 1:
        raise SystemExit(f"Expected one preprocessed dataset, found {dataset_dirs}")
    destination = dataset_dirs[0] / "splits_final.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(destination)


if __name__ == "__main__":
    main()

