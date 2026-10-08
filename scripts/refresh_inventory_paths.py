#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import MODALITIES
from tumortrust_vlm.data.inventory import _resolve_file
from tumortrust_vlm.utils import atomic_json_dump, sha256_json


def main() -> None:
    config = load_config("configs/base.yaml")
    path = ROOT / config["data"]["inventory"]
    records = json.loads(path.read_text(encoding="utf-8"))
    changes = []
    for record in records:
        existing = next((value for value in record["sequences"].values() if value), None)
        if existing is None:
            continue
        folder = Path(existing).parent
        for modality in MODALITIES:
            resolved = _resolve_file(folder, record["subject_id"], modality)
            new_value = str(resolved) if resolved else None
            if new_value != record["sequences"].get(modality):
                changes.append(
                    {
                        "subject_id": record["subject_id"],
                        "modality": modality,
                        "rejected": record["sequences"].get(modality),
                        "selected": new_value,
                    }
                )
                record["sequences"][modality] = new_value
        unreadable = [modality for modality in MODALITIES if not record["sequences"].get(modality)]
        if unreadable:
            record["complete_images"] = False
            record["eligible_for_split"] = False
            record["data_integrity_resolution"] = "excluded_unreadable_modality"
            record["unreadable_modalities"] = unreadable
    atomic_json_dump(records, path)
    detail = {"changes": changes, "inventory_sha256": sha256_json(records)}
    atomic_json_dump(detail, ROOT / "artifacts" / "private" / "path_resolution_changes.json")
    atomic_json_dump(
        {"changed_paths": len(changes), "inventory_sha256": detail["inventory_sha256"]},
        ROOT / "artifacts" / "path_resolution_summary.json",
    )
    print(json.dumps(detail, indent=2))


if __name__ == "__main__":
    main()
