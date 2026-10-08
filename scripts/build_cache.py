#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.cache import build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config["data"].get("cache_dir"):
        raise SystemExit("data.cache_dir is disabled")
    records = json.loads((ROOT / config["data"]["inventory"]).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8"))
    splits = set(args.splits or config["data"].get("cache_splits", ("train", "val")))
    allowed = {entry["subject_id"] for entry in manifest["entries"] if entry["split"] in splits}
    records = [record for record in records if record["subject_id"] in allowed]
    if args.limit is not None:
        records = records[: args.limit]
    if "test" in splits:
        print("WARNING: test data caching was explicitly requested; this is forbidden before final freeze.")
    print(json.dumps(build_cache(records, config, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
