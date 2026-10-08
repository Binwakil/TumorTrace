#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.inference import evidence_vector
from tumortrust_vlm.reporting.composition import compose_reporter_features
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def load_many(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        records.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose segmentation-only and classification-only OOF reporter features."
    )
    parser.add_argument("--segmentation", nargs="+", required=True)
    parser.add_argument("--classification", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--disable-referral", action="store_true")
    args = parser.parse_args()

    records = compose_reporter_features(
        load_many(args.segmentation), load_many(args.classification)
    )
    extraction_splits = {
        record.get("subject_provenance", {}).get("feature_extraction_split")
        for record in records
    }
    locked_test_opened = extraction_splits == {"test"}
    if locked_test_opened and not args.unlock_final_test:
        raise SystemExit("Final composed reporter features are locked")
    if args.disable_referral:
        for record in records:
            evidence = record["evidence"]
            evidence["referral"] = False
            evidence["referral_reasons"] = []
            evidence["unavailable_fields"] = sorted(
                set(evidence.get("unavailable_fields", [])) | {"referral"}
            )
            record["evidence_vector"] = evidence_vector(evidence)
    atomic_json_dump(records, args.output)
    summary = {
        "subjects": len(records),
        "output": args.output,
        "output_sha256": sha256_file(args.output),
        "segmentation_inputs": {
            path: sha256_file(path) for path in args.segmentation
        },
        "classification_inputs": {
            path: sha256_file(path) for path in args.classification
        },
        "feature_extraction_splits": sorted(str(value) for value in extraction_splits),
        "referral_disabled": args.disable_referral,
        "locked_test_opened": locked_test_opened,
    }
    atomic_json_dump(summary, Path(args.output).with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
