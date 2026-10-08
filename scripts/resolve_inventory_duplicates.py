#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.inventory import resolve_content_duplicates
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def main() -> None:
    config = load_config("configs/base.yaml")
    path = ROOT / config["data"]["inventory"]
    records = json.loads(path.read_text(encoding="utf-8"))
    resolutions = resolve_content_duplicates(records)
    atomic_json_dump(records, path)
    detailed_path = ROOT / "artifacts" / "private" / "duplicate_resolution.json"
    atomic_json_dump(resolutions, detailed_path)
    summary = {
        "groups": len(resolutions),
        "actions": dict(Counter(item["action"] for item in resolutions)),
        "eligible_labeled": sum(record.get("eligible_for_split", False) for record in records),
        "inventory_sha256_after_resolution": sha256_json(records),
    }
    atomic_json_dump(summary, ROOT / "artifacts" / "duplicate_resolution_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

