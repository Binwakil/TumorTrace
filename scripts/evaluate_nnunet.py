#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
from brats_evaluation import config_path, evaluate_single_exam
from panoptica import Panoptica_Evaluator

from tumortrust_vlm.evaluation.metrics import aggregate_case_metrics, segmentation_metrics
from tumortrust_vlm.evaluation.stratification import stratified_case_metrics
from tumortrust_vlm.evaluation.volumetry import aggregate_volume_metrics, case_volume_metrics
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

OFFICIAL_CONFIGS = {"GLI": "gli", "MEN": "MenPre", "MET": "mets"}


def validate_prediction_set(
    predictions: Path, records: list[dict], require_exact: bool = True
) -> dict:
    expected = {f"{record['case_id']}.nii.gz" for record in records}
    observed = {path.name for path in predictions.glob("*.nii.gz")}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or (require_exact and unexpected):
        raise ValueError(
            "nnU-Net prediction set does not match frozen validation cases: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return {
        "expected_predictions": len(expected),
        "observed_predictions": len(observed),
        "prediction_set_exact": not missing and not unexpected,
    }


def aggregate_official(records: list[dict]) -> dict:
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        cohort = record["cohort"]
        for region, payload in record["metrics"].items():
            if not isinstance(payload, dict):
                continue
            for metric, value in payload.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values[cohort][region][metric].append(float(value))
    return {
        cohort: {
            region: {
                metric: float(np.mean(metric_values))
                for metric, metric_values in sorted(metrics.items())
                if metric_values
            }
            for region, metrics in sorted(regions.items())
        }
        for cohort, regions in sorted(values.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net predictions on one explicitly selected TumorTrust split."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--raw-dataset",
        default="artifacts/private/nnunet_raw/Dataset501_TumorTrust",
    )
    parser.add_argument(
        "--mapping", default="artifacts/private/nnunet_501_mapping.json"
    )
    parser.add_argument(
        "--inventory", default="artifacts/private/master_inventory.json"
    )
    parser.add_argument(
        "--case-output", default="artifacts/private/nnunet_val_predictions.json"
    )
    parser.add_argument(
        "--official-output", default="artifacts/private/nnunet_val_official.json"
    )
    parser.add_argument(
        "--summary-output", default="artifacts/nnunet_val_summary.json"
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()

    predictions = Path(args.predictions)
    raw_dataset = Path(args.raw_dataset)
    mapping_path = Path(args.mapping)
    inventory_path = Path(args.inventory)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    metadata = {record["subject_id"]: record for record in inventory}
    if args.split == "test" and not args.unlock_final_test:
        raise SystemExit("Final nnU-Net test evaluation is locked")
    if args.split == "val" and any(record["split"] == "test" for record in mapping):
        raise SystemExit("nnU-Net mapping unexpectedly contains locked test subjects")
    validation = [record for record in mapping if record["split"] == args.split]
    expected_count = 248 if args.split == "val" else 320
    if len(validation) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {args.split} records, found {len(validation)}"
        )
    if args.limit is not None:
        validation = validation[: args.limit]
    prediction_audit = validate_prediction_set(
        predictions, validation, require_exact=args.limit is None
    )

    evaluators = (
        {
            cohort: Panoptica_Evaluator.load_from_config(str(config_path(name)))
            for cohort, name in OFFICIAL_CONFIGS.items()
        }
        if not args.skip_official
        else {}
    )
    cases = []
    official_records = []
    for record in validation:
        prediction_path = predictions / f"{record['case_id']}.nii.gz"
        label_directory = "labelsTr" if args.split == "val" else "labelsTs"
        reference_path = raw_dataset / label_directory / f"{record['case_id']}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        prediction_image = nib.load(prediction_path)
        reference_image = nib.load(reference_path)
        prediction = np.asarray(prediction_image.dataobj, dtype=np.uint8)
        reference = np.asarray(reference_image.dataobj, dtype=np.uint8)
        if prediction.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch for {record['case_id']}: "
                f"{prediction.shape} versus {reference.shape}"
            )
        spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
        subject = metadata[record["subject_id"]]
        case = {
            **segmentation_metrics(prediction, reference, spacing),
            **case_volume_metrics(prediction, reference, spacing),
            "subject_id": record["subject_id"],
            "cohort": subject["cohort"],
            "source_branch": subject["source_branch"],
            "source_orientation": subject["orientation"],
            "has_report": subject["has_report"],
        }
        cases.append(case)
        if evaluators:
            official_records.append(
                {
                    "subject_id": record["subject_id"],
                    "case_id": record["case_id"],
                    "cohort": subject["cohort"],
                    "metrics": evaluate_single_exam(
                        prediction_filepath=str(prediction_path),
                        reference_filepath=str(reference_path),
                        subject_identifier=record["case_id"],
                        evaluator=evaluators[subject["cohort"]],
                    ),
                }
            )

    aggregate = {
        **aggregate_case_metrics(cases),
        **aggregate_volume_metrics(cases),
        "subjects": len(cases),
        "stratified": stratified_case_metrics(cases),
        "official_brats": aggregate_official(official_records),
        "official_package": "BraTS-evaluation" if evaluators else None,
        "prediction_audit": prediction_audit,
        "mapping_sha256": sha256_file(mapping_path),
        "inventory_sha256": sha256_file(inventory_path),
        "evaluation_split": args.split,
        "locked_test_opened": args.split == "test" and args.unlock_final_test,
    }
    atomic_json_dump(cases, args.case_output)
    atomic_json_dump(official_records, args.official_output)
    atomic_json_dump(aggregate, args.summary_output)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
