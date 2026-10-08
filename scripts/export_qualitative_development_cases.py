#!/usr/bin/env python
"""Export frozen, prespecified development cases for manuscript visualization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import config_hash, load_config
from tumortrust_vlm.data.constants import COHORTS, MODALITIES
from tumortrust_vlm.data.splits import split_requires_final_unlock
from tumortrust_vlm.engine import build_dataset, build_model, load_checkpoint
from tumortrust_vlm.inference import infer_core_once
from tumortrust_vlm.reporting.evidence import build_evidence_card
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.utils import atomic_json_dump, sha256_file, sha256_json


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-config", default="outputs/D3_zscore/resolved_config.json")
    parser.add_argument("--seg-checkpoint", default="outputs/D3_zscore/checkpoints/best.pt")
    parser.add_argument("--seg-summary", default="outputs/D3_zscore/summary.json")
    parser.add_argument("--cls-config", default="outputs/C0/resolved_config.json")
    parser.add_argument("--cls-checkpoint", default="outputs/C0/checkpoints/best.pt")
    parser.add_argument(
        "--selection", default="artifacts/private/qualitative_case_selection.json"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/private/qualitative_development_cases"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seg_config = load_config(args.seg_config)
    cls_config = load_config(args.cls_config)
    if seg_config["data"]["split_manifest"] != cls_config["data"]["split_manifest"]:
        raise ValueError("Segmentation and classification configs do not share the frozen split")
    manifest_path = ROOT / seg_config["data"]["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split_requires_final_unlock(manifest, "val"):
        raise RuntimeError("Development validation unexpectedly requires final-test unlock")
    final_ids = {
        entry["subject_id"] for entry in manifest["entries"] if entry["split"] == "test"
    }

    selection = json.loads((ROOT / args.selection).read_text(encoding="utf-8"))
    if selection.get("locked_test_opened") is not False:
        raise ValueError("Qualitative selection lacks an explicit locked-test guard")
    selected = {record["subject_id"]: record for record in selection["cases"]}
    overlap = selected.keys() & final_ids
    if overlap:
        raise ValueError(f"Qualitative selection overlaps final test: {sorted(overlap)[:3]}")

    device = torch.device(args.device)
    seg_model = build_model(seg_config).to(device).eval()
    cls_model = build_model(cls_config).to(device).eval()
    seg_checkpoint = load_checkpoint(seg_model, args.seg_checkpoint, device)
    cls_checkpoint = load_checkpoint(cls_model, args.cls_checkpoint, device)
    seg_summary = json.loads((ROOT / args.seg_summary).read_text(encoding="utf-8"))
    checkpoint_config_hash = seg_checkpoint.get("config_hash")
    # The historical D3 resolved config was augmented for full-panel inference after
    # training, so its present hash differs from the immutable hash in both the
    # checkpoint and training summary. Require that independent archived agreement
    # instead of silently weakening the provenance check.
    if checkpoint_config_hash != seg_summary.get("config_hash"):
        raise ValueError("Segmentation checkpoint/training-summary mismatch")
    if cls_checkpoint.get("config_hash") != config_hash(cls_config):
        raise ValueError("Classification checkpoint/config mismatch")

    dataset = build_dataset(seg_config, "val", False)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for dataset_index, metadata in enumerate(dataset.records):
        subject_id = metadata["subject_id"]
        if subject_id not in selected:
            continue
        item = dataset[dataset_index]
        image = item["image"].unsqueeze(0).to(device)
        presence = item["modality_presence"].unsqueeze(0).to(device)
        seg_logits, _, _ = infer_core_once(
            seg_model,
            image,
            presence,
            roi_size=tuple(int(value) for value in seg_config["data"]["patch_size"]),
            sw_batch_size=int(seg_config["inference"].get("sw_batch_size", 1)),
            overlap=float(seg_config["inference"].get("sliding_window_overlap", 0.5)),
        )
        _, classification_latent = cls_model.encode_for_reporting_and_classification(
            image
        )
        cls_logits = cls_model.classify_latent(
            classification_latent, presence, image.shape[1]
        )
        probabilities = seg_logits.softmax(1)
        prediction = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
        predictive_entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(1)[0]
        brain = image.abs().sum(1)[0] > 0
        segmentation_uncertainty = float(
            torch.quantile(predictive_entropy[brain], 0.95)
            if brain.any()
            else predictive_entropy.mean()
        )
        class_probability = cls_logits.softmax(1)[0].cpu().numpy()
        class_probabilities = {
            cohort: float(class_probability[index])
            for index, cohort in enumerate(COHORTS)
        }
        classification_uncertainty = float(
            -np.sum(class_probability * np.log(np.clip(class_probability, 1e-8, 1)))
        )
        spacing = tuple(float(value) for value in item["spacing"])
        card = build_evidence_card(
            subject_id,
            prediction,
            spacing,
            class_probabilities,
            segmentation_uncertainty=segmentation_uncertainty,
            classification_uncertainty=classification_uncertainty,
        )
        selection_record = selected[subject_id]
        alias = selection_record["alias"]
        np.savez_compressed(
            output_dir / f"{alias}.npz",
            image=item["image"].cpu().numpy().astype(np.float16),
            target=item["label"].cpu().numpy().astype(np.uint8),
            prediction=prediction,
            predictive_entropy=predictive_entropy.cpu().numpy().astype(np.float16),
            spacing=np.asarray(spacing, dtype=np.float32),
            modalities=np.asarray(MODALITIES),
        )
        index.append(
            {
                **selection_record,
                "data_file": f"{alias}.npz",
                "evidence": card.to_dict(),
                "rendered_findings": render_findings(card),
            }
        )
    if set(selected) != {record["subject_id"] for record in index}:
        missing = sorted(set(selected) - {record["subject_id"] for record in index})
        raise ValueError(f"Selected development cases missing from dataset: {missing}")
    atomic_json_dump(
        {
            "scope": "development validation only",
            "selection_sha256": sha256_file(ROOT / args.selection),
            "segmentation_checkpoint_sha256": sha256_file(args.seg_checkpoint),
            "segmentation_checkpoint_config_hash": checkpoint_config_hash,
            "segmentation_current_config_hash": config_hash(seg_config),
            "segmentation_summary_sha256": sha256_file(ROOT / args.seg_summary),
            "classification_checkpoint_sha256": sha256_file(args.cls_checkpoint),
            "split_manifest_sha256": manifest.get("manifest_sha256", sha256_json(manifest)),
            "locked_test_opened": False,
            "cases": sorted(index, key=lambda record: record["alias"]),
        },
        output_dir / "index.json",
    )
    print(json.dumps({"subjects": len(index), "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
