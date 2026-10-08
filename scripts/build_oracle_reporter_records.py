#!/usr/bin/env python
"""Build development-only oracle-evidence records for the R5 diagnostic upper bound."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import COHORTS
from tumortrust_vlm.engine import build_dataset
from tumortrust_vlm.inference import evidence_vector
from tumortrust_vlm.reporting.dataset import validate_reporter_record_sets
from tumortrust_vlm.reporting.evidence import build_evidence_card
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="outputs/D3_zscore/resolved_config.json")
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--val-records", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-val", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = ROOT / config["data"]["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_ids = {entry["subject_id"] for entry in manifest["entries"] if entry["split"] == "test"}
    inputs = {
        "train": json.loads(Path(args.train_records).read_text(encoding="utf-8")),
        "val": json.loads(Path(args.val_records).read_text(encoding="utf-8")),
    }
    requested = {record["subject_id"] for records in inputs.values() for record in records}
    if requested & final_ids:
        raise RuntimeError("Oracle diagnostic records overlap the locked final test")

    locations: dict[str, tuple[object, int]] = {}
    for split in ("train", "val"):
        dataset = build_dataset(config, split, False)
        for index, metadata in enumerate(dataset.records):
            subject_id = metadata["subject_id"]
            if subject_id in requested:
                if subject_id in locations:
                    raise ValueError(f"Duplicate development subject: {subject_id}")
                locations[subject_id] = (dataset, index)
    if requested != set(locations):
        raise ValueError(f"Missing development labels: {sorted(requested - set(locations))[:5]}")

    outputs: dict[str, list[dict]] = {"train": [], "val": []}
    for report_split, records in inputs.items():
        for source_record in records:
            record = copy.deepcopy(source_record)
            dataset, index = locations[record["subject_id"]]
            item = dataset[index]
            target = np.squeeze(item["label"].cpu().numpy()).astype(np.uint8)
            spacing = tuple(float(value) for value in item["spacing"])
            probabilities = {cohort: float(cohort == record["cohort"]) for cohort in COHORTS}
            card = build_evidence_card(
                record["subject_id"],
                target,
                spacing,
                probabilities,
                segmentation_uncertainty=0.0,
                classification_uncertainty=0.0,
            ).to_dict()
            record["evidence"] = card
            record["evidence_vector"] = evidence_vector(card)
            record["oracle_evidence"] = True
            record["oracle_evidence_scope"] = "development-only diagnostic upper bound"
            outputs[report_split].append(record)

    audit = validate_reporter_record_sets(outputs["train"], outputs["val"])
    atomic_json_dump(outputs["train"], args.output_train)
    atomic_json_dump(outputs["val"], args.output_val)
    print(
        json.dumps(
            {
                **audit,
                "input_train_sha256": sha256_file(args.train_records),
                "input_val_sha256": sha256_file(args.val_records),
                "locked_test_opened": False,
                "output_train": args.output_train,
                "output_val": args.output_val,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
