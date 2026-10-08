#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tumortrust_vlm.ablation import changed_config_fields
from tumortrust_vlm.config import config_hash, load_config
from tumortrust_vlm.data.constants import COHORTS
from tumortrust_vlm.data.dataset import collate_training
from tumortrust_vlm.data.splits import split_requires_final_unlock
from tumortrust_vlm.engine import build_dataset, build_model, load_checkpoint
from tumortrust_vlm.evaluation.metrics import dice_score, region_mask
from tumortrust_vlm.inference import evidence_vector, infer_core_once
from tumortrust_vlm.models.reporter import RegionAwarePooler
from tumortrust_vlm.reporting.evidence import build_evidence_card, volumes_from_mask
from tumortrust_vlm.utils import atomic_json_dump, sha256_file, sha256_json


def checkpoint_config_differences(reference: dict, candidate: dict) -> set[str]:
    changed = changed_config_fields(reference, candidate)
    # Legacy resolved JSON files parse exponent-only JSON numbers (for example ``1e-05``)
    # as strings through PyYAML, while the original frozen YAML parses ``1.0e-05`` as a
    # float. Training casts this field to float before constructing AdamW, so this is a
    # serialization-only difference rather than a hyperparameter change.
    if "training.weight_decay" in changed and float(reference["training"]["weight_decay"]) == float(
        candidate["training"]["weight_decay"]
    ):
        changed.remove("training.weight_decay")
    return changed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument(
        "--checkpoint-config",
        nargs="+",
        help="Resolved config for each checkpoint; defaults to --config for one-model extraction.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--calibration")
    parser.add_argument("--uncertainty-map-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_configs = (
        [load_config(path) for path in args.checkpoint_config]
        if args.checkpoint_config
        else [config] * len(args.checkpoint)
    )
    if len(checkpoint_configs) != len(args.checkpoint):
        raise ValueError("--checkpoint-config must provide one config per checkpoint")
    reference_checkpoint_config = checkpoint_configs[0]
    for index, candidate in enumerate(checkpoint_configs[1:], start=1):
        changed = checkpoint_config_differences(reference_checkpoint_config, candidate)
        if changed - {"project.seed"}:
            raise ValueError(
                f"Ensemble checkpoint config {index} is not seed-matched: {sorted(changed)}"
            )
    manifest_path = Path(config["data"]["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split_requires_final_unlock(manifest, args.split) and not args.unlock_final_test:
        raise SystemExit("Final test features are locked")
    calibration = (
        json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else {}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    checkpoint_config_hashes = []
    for checkpoint_path, checkpoint_config in zip(args.checkpoint, checkpoint_configs, strict=True):
        model = build_model(checkpoint_config).to(device).eval()
        checkpoint = load_checkpoint(model, checkpoint_path, device)
        expected_hash = config_hash(checkpoint_config)
        if checkpoint.get("config_hash") != expected_hash:
            raise ValueError(f"Checkpoint/config mismatch: {checkpoint_path}")
        checkpoint_config_hashes.append(expected_hash)
        models.append(model)
    checkpoint_hashes = [sha256_file(path) for path in args.checkpoint]
    training_subject_ids = sorted(
        entry["subject_id"] for entry in manifest["entries"] if entry["split"] == "train"
    )
    training_subject_set = set(training_subject_ids)
    model_provenance = {
        "ensemble_checkpoints": len(models),
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_config_sha256": checkpoint_config_hashes,
        "inference_config_sha256": config_hash(config),
        "split_manifest_sha256": manifest.get("manifest_sha256", sha256_json(manifest)),
        "training_subjects_sha256": sha256_json(training_subject_ids),
        "mc_samples_per_checkpoint": args.mc_samples,
    }
    if args.mc_samples < 1:
        raise ValueError("--mc-samples must be at least 1")
    dataset = build_dataset(config, args.split, False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_training
    )
    pooler = RegionAwarePooler().to(device)
    records = []
    probe_features = []
    probe_cohort = []
    probe_source = []
    for index, batch in enumerate(loader):
        if args.max_cases is not None and index >= args.max_cases:
            break
        image = batch["image"].to(device)
        presence = batch["modality_presence"].to(device)
        if args.mc_samples > 1:
            for model in models:
                for module in model.modules():
                    if isinstance(
                        module,
                        (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d),
                    ):
                        module.train()
        probability_sum = None
        expected_voxel_entropy_sum = None
        class_probability_sum = None
        class_expected_entropy_sum = None
        latent_sum = None
        sample_count = 0
        temperature = float(calibration.get("temperature", 1.0))
        roi_size = tuple(int(value) for value in config["data"]["patch_size"])
        for model in models:
            for _ in range(args.mc_samples):
                segmentation_logits, classification_logits, reporting_latent = infer_core_once(
                    model,
                    image,
                    presence,
                    roi_size=roi_size,
                    sw_batch_size=int(config["inference"].get("sw_batch_size", 1)),
                    overlap=float(config["inference"].get("sliding_window_overlap", 0.5)),
                )
                sample_probability = segmentation_logits.softmax(1)
                sample_voxel_entropy = -(
                    sample_probability * sample_probability.clamp_min(1e-8).log()
                ).sum(1)
                sample_class_probability = (classification_logits / temperature).softmax(1)
                sample_class_entropy = -(
                    sample_class_probability * sample_class_probability.clamp_min(1e-8).log()
                ).sum(1)
                probability_sum = (
                    sample_probability
                    if probability_sum is None
                    else probability_sum + sample_probability
                )
                expected_voxel_entropy_sum = (
                    sample_voxel_entropy
                    if expected_voxel_entropy_sum is None
                    else expected_voxel_entropy_sum + sample_voxel_entropy
                )
                class_probability_sum = (
                    sample_class_probability
                    if class_probability_sum is None
                    else class_probability_sum + sample_class_probability
                )
                class_expected_entropy_sum = (
                    sample_class_entropy
                    if class_expected_entropy_sum is None
                    else class_expected_entropy_sum + sample_class_entropy
                )
                latent_sum = (
                    reporting_latent if latent_sum is None else latent_sum + reporting_latent
                )
                sample_count += 1
        assert probability_sum is not None
        assert expected_voxel_entropy_sum is not None
        assert class_probability_sum is not None
        assert class_expected_entropy_sum is not None
        assert latent_sum is not None
        probabilities = probability_sum / sample_count
        prediction = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
        class_probability_tensor = class_probability_sum / sample_count
        class_probability = class_probability_tensor[0].cpu().numpy()
        voxel_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)[0]
        expected_voxel_entropy = (expected_voxel_entropy_sum / sample_count)[0]
        voxel_mutual_information = voxel_entropy - expected_voxel_entropy
        brain = image.abs().sum(1)[0] > 0
        segmentation_uncertainty = (
            float(torch.quantile(voxel_entropy[brain], 0.95))
            if brain.any()
            else float(voxel_entropy.mean())
        )
        segmentation_mutual_information = (
            float(torch.quantile(voxel_mutual_information[brain], 0.95))
            if brain.any()
            else float(voxel_mutual_information.mean())
        )
        classification_uncertainty = float(
            -np.sum(class_probability * np.log(np.clip(class_probability, 1e-8, 1)))
        )
        class_sample_entropy = (class_expected_entropy_sum / sample_count).mean()
        classification_mutual_information = float(
            classification_uncertainty - class_sample_entropy.cpu()
        )
        class_probabilities = {
            cohort: float(class_probability[position]) for position, cohort in enumerate(COHORTS)
        }
        predicted_volumes = volumes_from_mask(
            prediction, tuple(float(value) for value in batch["spacing"][0])
        )
        volume_intervals_90 = {}
        volume_intervals_95 = {}
        for region, value in predicted_volumes.items():
            region_calibration = calibration.get("volume_intervals", {}).get(region, {})
            for coverage, destination in (
                ("0.9", volume_intervals_90),
                ("0.95", volume_intervals_95),
            ):
                payload = region_calibration.get(coverage)
                if payload:
                    quantile = float(payload["residual_quantile_ml"])
                    destination[region] = (max(0.0, value - quantile), value + quantile)
        segmentation_referral = calibration.get("segmentation_referral", {})
        referral_signal = segmentation_referral.get("signal", "predictive_entropy")
        if referral_signal == "predictive_entropy":
            card_segmentation_uncertainty = segmentation_uncertainty
        elif referral_signal == "mutual_information":
            card_segmentation_uncertainty = segmentation_mutual_information
        else:
            raise ValueError(f"Unsupported segmentation referral signal: {referral_signal}")
        card = build_evidence_card(
            batch["subject_id"][0],
            prediction,
            tuple(float(value) for value in batch["spacing"][0]),
            class_probabilities,
            volume_intervals_90=volume_intervals_90,
            volume_intervals_95=volume_intervals_95,
            segmentation_uncertainty=card_segmentation_uncertainty,
            classification_uncertainty=classification_uncertainty,
            segmentation_referral_threshold=segmentation_referral.get("uncertainty_threshold"),
            classification_referral_threshold=calibration.get("classification_referral", {}).get(
                "uncertainty_threshold"
            ),
        )
        mean_latent = latent_sum / sample_count
        region_tokens = pooler(mean_latent, probabilities).cpu().numpy()[0]
        card_dictionary = card.to_dict()
        target_array = batch["label"][0].cpu().numpy()
        case_dice = {
            region: dice_score(region_mask(prediction, region), region_mask(target_array, region))
            for region in ("WT", "TC", "ET", "SNFH")
        }
        if args.uncertainty_map_dir:
            map_root = Path(args.uncertainty_map_dir)
            map_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                map_root / f"{batch['subject_id'][0]}.npz",
                prediction=prediction,
                target=batch["label"].cpu().numpy()[0].astype(np.uint8),
                predictive_entropy=voxel_entropy.cpu().numpy().astype(np.float16),
                mutual_information=voxel_mutual_information.cpu().numpy().astype(np.float16),
                brain_mask=brain.cpu().numpy().astype(np.uint8),
            )
        records.append(
            {
                "subject_id": batch["subject_id"][0],
                "cohort": batch["cohort"][0],
                "source_branch": batch["source_branch"][0],
                "evidence": card_dictionary,
                "evidence_vector": evidence_vector(card_dictionary),
                "visual_tokens": region_tokens.tolist(),
                "global_features": mean_latent.mean((2, 3, 4)).cpu().numpy()[0].tolist(),
                "uncertainty": {
                    "segmentation_predictive_entropy_p95": segmentation_uncertainty,
                    "segmentation_mutual_information_p95": segmentation_mutual_information,
                    "classification_predictive_entropy": classification_uncertainty,
                    "classification_mutual_information": classification_mutual_information,
                    "samples": sample_count,
                },
                "segmentation_evaluation": {
                    **{f"dice_{region}": value for region, value in case_dice.items()},
                    "macro_dice": float(np.mean(list(case_dice.values()))),
                },
                "inference_provenance": {
                    **model_provenance,
                },
                "subject_provenance": {
                    "feature_extraction_split": args.split,
                    "excluded_from_checkpoint_training": (
                        batch["subject_id"][0] not in training_subject_set
                    ),
                },
            }
        )
        probe_features.append(mean_latent.mean((2, 3, 4)).cpu().numpy()[0])
        probe_cohort.append(COHORTS.index(batch["cohort"][0]))
        source_branch = dataset.records[index]["source_branch"]
        probe_source.append(source_branch)
    output = Path(args.output)
    atomic_json_dump(records, output)
    unique_sources = {name: index for index, name in enumerate(sorted(set(probe_source)))}
    np.savez_compressed(
        output.with_suffix(".probe.npz"),
        features=np.asarray(probe_features),
        cohort=np.asarray(probe_cohort),
        source=np.asarray([unique_sources[value] for value in probe_source]),
    )
    print(
        json.dumps(
            {
                "subjects": len(records),
                "output": str(output),
                "checkpoints": len(models),
                "samples_per_checkpoint": args.mc_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
