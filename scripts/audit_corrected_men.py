#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import REGION_LABELS
from tumortrust_vlm.evaluation.metrics import dice_score
from tumortrust_vlm.utils import atomic_json_dump


def resolve_nifti(directory: Path, stem: str) -> Path:
    candidates = (directory / f"{stem}.nii.gz", directory / f"{stem}.nii")
    available = [path for path in candidates if path.is_file()]
    if len(available) != 1:
        raise FileNotFoundError(f"Expected exactly one NIfTI for {directory / stem}")
    return available[0]


def compare_labels(original: np.ndarray, corrected: np.ndarray, voxel_volume_mm3: float) -> dict:
    original = np.asarray(original).astype(np.int16)
    corrected = np.asarray(corrected).astype(np.int16)
    original[original == 4] = 3
    corrected[corrected == 4] = 3
    if original.shape != corrected.shape:
        raise ValueError("Original and corrected labels have different shapes")
    union = (original > 0) | (corrected > 0)
    changed = original != corrected
    result = {
        "changed_voxels": int(changed.sum()),
        "changed_fraction_of_union": float(changed.sum() / max(1, union.sum())),
        "exactly_identical": bool(not changed.any()),
    }
    for region, labels in REGION_LABELS.items():
        original_region = np.isin(original, labels)
        corrected_region = np.isin(corrected, labels)
        result[f"dice_{region}"] = dice_score(original_region, corrected_region)
        original_ml = float(original_region.sum() * voxel_volume_mm3 / 1000)
        corrected_ml = float(corrected_region.sum() * voxel_volume_mm3 / 1000)
        result[f"original_volume_ml_{region}"] = original_ml
        result[f"corrected_volume_ml_{region}"] = corrected_ml
        result[f"absolute_volume_change_ml_{region}"] = abs(corrected_ml - original_ml)
    return result


def summarize(cases: list[dict]) -> dict:
    metrics = sorted(
        key
        for key, value in cases[0].items()
        if key != "subject_id" and isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return {
        "subjects": len(cases),
        "exactly_identical": sum(case["exactly_identical"] for case in cases),
        "metrics": {
            key: {
                "mean": float(np.mean([case[key] for case in cases])),
                "median": float(np.median([case[key] for case in cases])),
                "maximum": float(np.max([case[key] for case in cases])),
            }
            for key in metrics
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit original versus corrected MEN annotations without model evaluation."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset_root = Path(config["data"]["dataset_root"])
    original_root = dataset_root / "2023MEN" / "TrainingData"
    corrected_root = dataset_root / "2023MEN" / "BraTS-MEN-TRAIN-FIX-V4"
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    permitted = {
        entry["subject_id"] for entry in manifest["entries"] if entry["split"] == args.split
    }
    cases = []
    for corrected_subject in sorted(path for path in corrected_root.iterdir() if path.is_dir()):
        subject_id = corrected_subject.name
        if subject_id not in permitted:
            continue
        try:
            original_path = resolve_nifti(original_root / subject_id, f"{subject_id}-seg")
            corrected_path = resolve_nifti(corrected_subject, f"{subject_id}-seg")
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        original_image = nib.as_closest_canonical(nib.load(original_path))
        corrected_image = nib.as_closest_canonical(nib.load(corrected_path))
        if not np.allclose(original_image.affine, corrected_image.affine, atol=1e-4):
            raise SystemExit(f"Original/corrected affine mismatch for {subject_id}")
        voxel_volume = float(np.prod(original_image.header.get_zooms()[:3]))
        cases.append(
            {
                "subject_id": subject_id,
                **compare_labels(
                    np.asarray(original_image.dataobj),
                    np.asarray(corrected_image.dataobj),
                    voxel_volume,
                ),
            }
        )
    if not cases:
        raise SystemExit("No corrected MEN cases occur in the requested development split")
    result = {
        "split": args.split,
        "audit_scope": "annotation_pair_comparison_without_model_predictions",
        **summarize(cases),
    }
    atomic_json_dump(result, args.output)
    if args.case_output:
        atomic_json_dump(cases, args.case_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
