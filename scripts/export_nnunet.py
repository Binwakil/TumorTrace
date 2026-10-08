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
from tumortrust_vlm.utils import atomic_json_dump


def save_canonical(source: str, destination: Path, is_label: bool = False) -> None:
    image = nib.as_closest_canonical(nib.load(source))
    array = np.asarray(image.dataobj)
    if is_label:
        array = array.astype(np.uint8)
        array[array == 4] = 3
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(array, image.affine, image.header), temporary)
    os.replace(temporary, destination)


def export_case(arguments: tuple[dict, dict, str]) -> str:
    entry, record, output_string = arguments
    output = Path(output_string)
    case_id = entry["case_id"]
    for channel, modality in enumerate(MODALITIES):
        destination = output / "imagesTr" / f"{case_id}_{channel:04d}.nii.gz"
        if not destination.is_file():
            save_canonical(record["sequences"][modality], destination)
    label_destination = output / "labelsTr" / f"{case_id}.nii.gz"
    if not label_destination.is_file():
        save_canonical(record["segmentation"], label_destination, is_label=True)
    return case_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--output-root", default="artifacts/private/nnunet_raw")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    records = json.loads((ROOT / config["data"]["inventory"]).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["data"]["split_manifest"]).read_text(encoding="utf-8"))
    record_by_id = {record["subject_id"]: record for record in records}
    entries = [entry for entry in manifest["entries"] if entry["split"] in args.splits]
    if args.limit is not None:
        entries = entries[: args.limit]
    dataset_name = f"Dataset{args.dataset_id:03d}_TumorTrust"
    output = ROOT / args.output_root / dataset_name
    mapping = []
    split_cases = {"train": [], "val": []}
    jobs = []
    for index, entry in enumerate(entries):
        record = record_by_id[entry["subject_id"]]
        case_id = f"TT_{entry['cohort']}_{index:05d}"
        mapping.append({"case_id": case_id, "subject_id": entry["subject_id"], "split": entry["split"]})
        split_cases[entry["split"]].append(case_id)
        jobs.append(({**entry, "case_id": case_id}, record, str(output)))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(export_case, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 50 == 0:
                print(f"[nnU-Net export] {completed}/{len(entries)}", flush=True)
    dataset = {
        "channel_names": {"0": "T1", "1": "T1c", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "NCR_or_NETC": 1, "SNFH": 2, "ET": 3},
        "numTraining": len(entries),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    atomic_json_dump(dataset, output / "dataset.json")
    atomic_json_dump(mapping, ROOT / "artifacts" / "private" / f"nnunet_{args.dataset_id}_mapping.json")
    atomic_json_dump(
        [{"train": split_cases["train"], "val": split_cases["val"]}],
        ROOT / "artifacts" / "private" / f"nnunet_{args.dataset_id}_splits_final.json",
    )
    summary = {
        "dataset": dataset_name,
        "cases": len(entries),
        "train": len(split_cases["train"]),
        "val": len(split_cases["val"]),
        "raw_root_private": True,
        "test_exported": False,
    }
    atomic_json_dump(summary, ROOT / "artifacts" / f"nnunet_{args.dataset_id}_export_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
