#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.preprocessing import load_preprocessed_case
from tumortrust_vlm.reporting.evidence import laterality_from_canonical_mask, volumes_from_mask
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--per-stratum", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    records = json.loads((ROOT / config["data"]["inventory"]).read_text(encoding="utf-8"))
    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["complete_images"] and record["labeled"]:
            strata[(record["cohort"], record["source_branch"], record["orientation"])].append(record)
    cases = []
    for stratum in sorted(strata):
        for record in strata[stratum][: args.per_stratum]:
            image, mask, transform, presence = load_preprocessed_case(
                record,
                target_spacing=config["data"]["target_spacing"],
                normalization=config["data"]["normalization"],
                crop_margin=config["data"]["brain_crop_margin"],
                require_segmentation=True,
            )
            assert mask is not None
            cases.append(
                {
                    "subject_id": record["subject_id"],
                    "cohort": record["cohort"],
                    "branch": record["source_branch"],
                    "source_orientation": record["orientation"],
                    "image_shape": list(image.shape),
                    "mask_shape": list(mask.shape),
                    "finite": bool(np.isfinite(image).all()),
                    "modalities_present": presence.tolist(),
                    "label_values": [int(value) for value in np.unique(mask)],
                    "laterality_canonical": laterality_from_canonical_mask(mask),
                    "volumes_ml": volumes_from_mask(mask, transform.spacing),
                }
            )
    if not all(case["finite"] and case["image_shape"][1:] == case["mask_shape"] for case in cases):
        raise SystemExit("Data validation failed")
    aggregate = {"strata": len(strata), "cases_checked": len(cases), "all_passed": True, "cases": cases}
    atomic_json_dump(aggregate, ROOT / "artifacts" / "private" / "preprocessing_validation.json")
    atomic_json_dump(
        {"strata": len(strata), "cases_checked": len(cases), "all_passed": True},
        ROOT / "artifacts" / "preprocessing_validation_summary.json",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

