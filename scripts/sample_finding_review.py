#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import REPORT_DIRECTORIES
from tumortrust_vlm.data.inventory import load_report_splits
from tumortrust_vlm.reporting.findings import (
    FINDING_REVIEW_STATUS,
    FINDING_SCHEMA_VERSION,
    extract_normalized_findings,
)
from tumortrust_vlm.utils import atomic_json_dump, sha256_json

REVIEW_FIELDS = (
    "multiplicity",
    "lateralities",
    "anatomic_sites",
    "enhancement_presence",
    "enhancement_patterns",
    "enhancement_degree",
    "edema",
    "midline_shift",
    "mass_effect",
    "margin",
    "largest_reported_dimensions_mm",
)

INSTRUCTIONS = """# Normalized finding-field review

Review only what the supplied MRI narrative explicitly states. Do not infer a diagnosis or inspect
the model extraction. Use `unspecified` when the narrative does not support a value.

- `multiplicity`: `single`, `multiple`, or `unspecified`.
- `lateralities`: semicolon-separated `left`, `right`, `bilateral`, and/or `midline`; or
  `unspecified`. Do not use the direction of midline shift as lesion laterality.
- `anatomic_sites`: semicolon-separated controlled sites from the accompanying data dictionary;
  use `unspecified` if none apply. Allowed sites are `basal_ganglia`, `brainstem`, `cerebellum`,
  `corpus_callosum`, `frontal_lobe`, `insular_region`, `midline_fissure`, `occipital_lobe`,
  `parietal_lobe`, `sellar_region`, `skull_base`, `temporal_lobe`, `thalamus`, and `ventricular`.
- `enhancement_presence`: `present`, `absent`, or `unspecified`.
- `enhancement_patterns`: semicolon-separated `ring`, `peripheral`, `heterogeneous`, and/or
  `homogeneous`; or `unspecified`.
- `enhancement_degree`: `mild`, `moderate`, `marked`, or `unspecified`.
- `edema`: `absent`, `mild`, `extensive`, `present_unspecified`, or `unspecified`.
- `midline_shift`: `absent`, `present_leftward`, `present_rightward`,
  `present_unspecified`, or `unspecified`.
- `mass_effect`: `present`, `absent`, or `unspecified`.
- `margin`: `ill_defined`, `well_defined`, or `unspecified`.
- `largest_reported_dimensions_mm`: three numbers as `A*B*C`, or `unspecified`.

Set `review_complete` to `yes` only after every field is filled. Add free-text concerns in
`review_notes`. The packet contains development reports only and must remain in the private
artifact directory.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded development-report review packet.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--per-cohort", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", default="artifacts/private/finding_review")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligible = {entry["subject_id"] for entry in manifest["entries"]}
    report_splits = load_report_splits(config["data"]["report_split"])
    rng = random.Random(args.seed)
    selected = []
    for cohort, directory in REPORT_DIRECTORIES.items():
        path = Path(config["data"]["report_meta"]) / directory / "global_finding.json"
        reports = json.loads(path.read_text(encoding="utf-8"))
        candidates = [
            (subject_id, text)
            for subject_id, text in reports.items()
            if subject_id in eligible and report_splits.get(subject_id) in {"train", "val"}
        ]
        if len(candidates) < args.per_cohort:
            raise ValueError(f"Only {len(candidates)} eligible {cohort} development reports")
        rng.shuffle(candidates)
        selected.extend((cohort, *record) for record in candidates[: args.per_cohort])
    rng.shuffle(selected)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    review_path = output / "blind_review.csv"
    fieldnames = [
        "case_code",
        "report_text",
        *(f"reviewer_{field}" for field in REVIEW_FIELDS),
        "review_complete",
        "review_notes",
    ]
    private_reference = []
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (cohort, subject_id, text) in enumerate(selected, start=1):
            case_code = f"FFR-{index:03d}"
            writer.writerow({"case_code": case_code, "report_text": text})
            private_reference.append(
                {
                    "case_code": case_code,
                    "subject_id": subject_id,
                    "cohort": cohort,
                    "report_split": report_splits[subject_id],
                    "automatic_findings": extract_normalized_findings(text),
                }
            )
    atomic_json_dump(private_reference, output / "private_reference.json")
    (output / "REVIEW_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    summary = {
        "schema_version": FINDING_SCHEMA_VERSION,
        "review_status": FINDING_REVIEW_STATUS,
        "seed": args.seed,
        "per_cohort": args.per_cohort,
        "subjects": len(selected),
        "development_splits_only": True,
        "selection_sha256": sha256_json(
            [(record["case_code"], record["subject_id"]) for record in private_reference]
        ),
        "review_packet": str(review_path),
    }
    atomic_json_dump(summary, output / "sampling_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
