#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import MODALITIES
from tumortrust_vlm.utils import atomic_json_dump, sha256_file, sha256_json


def save_canonical(source: str, destination: Path, *, label: bool = False) -> None:
    image = nib.as_closest_canonical(nib.load(source))
    array = np.asarray(image.dataobj)
    if label:
        array = array.astype(np.uint8)
        array[array == 4] = 3
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(array, image.affine, image.header), temporary)
    os.replace(temporary, destination)


def export_case(job: tuple[dict, dict, str]) -> str:
    entry, inventory_record, output_value = job
    output = Path(output_value)
    case_id = entry["case_id"]
    for channel, modality in enumerate(MODALITIES):
        destination = output / "imagesTs" / f"{case_id}_{channel:04d}.nii.gz"
        if not destination.is_file():
            save_canonical(inventory_record["sequences"][modality], destination)
    label_destination = output / "labelsTs" / f"{case_id}.nii.gz"
    if not label_destination.is_file():
        save_canonical(inventory_record["segmentation"], label_destination, label=True)
    return case_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot export of the locked 320-case test for frozen nnU-Net inference."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--output", default="artifacts/private/nnunet_final_test/Dataset501_TumorTrustFinalTest"
    )
    parser.add_argument(
        "--mapping", default="artifacts/private/nnunet_final_test_mapping.json"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_final_test:
        raise SystemExit("Final nnU-Net export is locked")

    config_path = ROOT / args.config
    config = load_config(config_path)
    inventory_path = ROOT / config["data"]["inventory"]
    manifest_path = ROOT / config["data"]["split_manifest"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory_by_id = {record["subject_id"]: record for record in inventory}
    test_entries = sorted(
        (entry for entry in manifest["entries"] if entry["split"] == "test"),
        key=lambda entry: entry["subject_id"],
    )
    if len(test_entries) != 320:
        raise ValueError(f"Expected the locked 320-case test, found {len(test_entries)}")
    output = ROOT / args.output
    mapping = []
    jobs = []
    for index, entry in enumerate(test_entries):
        case_id = f"TT_FINAL_{entry['cohort']}_{index:05d}"
        mapping.append(
            {"case_id": case_id, "subject_id": entry["subject_id"], "split": "test"}
        )
        jobs.append(
            (
                {**entry, "case_id": case_id},
                inventory_by_id[entry["subject_id"]],
                str(output),
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(export_case, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 50 == 0:
                print(f"[locked nnU-Net export] {completed}/{len(jobs)}", flush=True)

    mapping_path = ROOT / args.mapping
    atomic_json_dump(mapping, mapping_path)
    summary = {
        "subjects": len(mapping),
        "split": "test",
        "output": str(output),
        "mapping": args.mapping,
        "mapping_sha256": sha256_file(mapping_path),
        "subject_ids_sha256": sha256_json(sorted(row["subject_id"] for row in mapping)),
        "master_manifest_sha256": sha256_file(manifest_path),
        "locked_test_opened": True,
    }
    atomic_json_dump(summary, ROOT / "artifacts/nnunet_final_test_export_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
