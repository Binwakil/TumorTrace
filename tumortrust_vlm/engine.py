from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from tumortrust_vlm.config import config_hash
from tumortrust_vlm.data.dataset import TumorTrustDataset, collate_training
from tumortrust_vlm.evaluation.metrics import (
    aggregate_case_metrics,
    classification_metrics,
    segmentation_metrics,
)
from tumortrust_vlm.evaluation.stratification import stratified_case_metrics
from tumortrust_vlm.evaluation.volumetry import aggregate_volume_metrics, case_volume_metrics
from tumortrust_vlm.models.core import (
    MultiTaskSegResNet,
    SeparateEncoderMultiTaskSegResNet,
)
from tumortrust_vlm.models.losses import (
    MultiTaskObjective,
    gradient_diagnostics,
    pcgrad_backward,
)
from tumortrust_vlm.utils import (
    atomic_json_dump,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def build_model(config: dict[str, Any]) -> MultiTaskSegResNet:
    model_config = config["model"]
    architectures = {
        "multitask_segresnet": MultiTaskSegResNet,
        "separate_encoder_segresnet": SeparateEncoderMultiTaskSegResNet,
    }
    architecture = model_config.get("architecture", "multitask_segresnet")
    if architecture not in architectures:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return architectures[architecture](
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        classification_classes=model_config["classification_classes"],
        init_filters=model_config["init_filters"],
        dropout=model_config["dropout"],
    )


def build_dataset(
    config: dict[str, Any], split: str, training: bool, patch_validation: bool = False
) -> TumorTrustDataset:
    data = config["data"]
    patch_size = data["patch_size"] if training or patch_validation else None
    return TumorTrustDataset(
        data["inventory"],
        data["split_manifest"],
        split=split,
        patch_size=patch_size,
        training=training,
        seed=config["project"]["seed"],
        normalization=data.get("normalization", "robust_nonzero"),
        target_spacing=data.get("target_spacing", (1, 1, 1)),
        crop_margin=data.get("brain_crop_margin", 8),
        positive_probability=config["training"].get("positive_patch_probability", 0.75),
        canonicalize=data.get("canonicalize", True),
        input_view=data.get("input_view", "whole"),
        missing_modality_probability=data.get("missing_modality_probability", 0.0),
        force_missing_modality=data.get("force_missing_modality"),
        cache_dir=data.get("cache_dir"),
        cache_config=config,
        label_mode=data.get("label_mode", "cohort_native_4class"),
        robustness_shift=data.get("robustness_shift", "none"),
    )


def stratified_validation_panel(records: list[dict], maximum: int | None, seed: int) -> list[dict]:
    if maximum is None or maximum >= len(records):
        return records
    if maximum < 1:
        raise ValueError("Development validation panel size must be positive")
    grouped: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["cohort"], record["source_branch"], bool(record.get("has_report")))].append(
            record
        )
    rng = random.Random(seed)
    for group in grouped.values():
        group.sort(key=lambda item: item["subject_id"])
        rng.shuffle(group)
    selected: list[dict] = []
    keys = sorted(grouped)
    while len(selected) < maximum:
        added = False
        for key in keys:
            if grouped[key] and len(selected) < maximum:
                selected.append(grouped[key].pop())
                added = True
        if not added:
            break
    return selected


def development_validation_limit(config: dict[str, Any]) -> int | None:
    explicit = config["training"].get("max_validation_cases")
    if explicit is not None:
        return int(explicit)
    if config["model"].get("joint_weighting") == "classification_only":
        return None
    if config["training"].get("full_validation_during_training", False):
        return None
    return 24


def _segmentation_predictor(model: MultiTaskSegResNet, presence: torch.Tensor):
    def predict(image: torch.Tensor) -> torch.Tensor:
        return model.segment(image)[0]

    return predict


def _epochs_since_best(history: list[dict], metric: str) -> int:
    """Recover early-stopping progress from legacy checkpoints without stored state."""
    scored = [record for record in history if f"val_{metric}" in record]
    if not scored:
        return 0
    best_record = max(scored, key=lambda record: record[f"val_{metric}"])
    return int(history[-1]["epoch"]) - int(best_record["epoch"])


@torch.no_grad()
def evaluate_model(
    model: MultiTaskSegResNet,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    *,
    patch_validation: bool = False,
    max_cases: int | None = None,
    official_export_dir: str | Path | None = None,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    case_results: list[dict] = []
    all_probabilities: list[np.ndarray] = []
    all_labels: list[int] = []
    roi = tuple(config["data"]["patch_size"])
    overlap = float(config["inference"].get("sliding_window_overlap", 0.5))
    objective_mode = config["model"].get("joint_weighting")
    classification_only = objective_mode == "classification_only"
    segmentation_only = objective_mode == "segmentation_only"
    for index, batch in enumerate(loader):
        if max_cases is not None and index >= max_cases:
            break
        image = batch["image"].to(device)
        presence = batch["modality_presence"].to(device)
        classification_logits = None
        if classification_only:
            classification_latent = model.encode_for_classification(image)
            classification_logits = model.classify_latent(
                classification_latent, presence, image.shape[1]
            )
        elif patch_validation or all(image.shape[axis + 2] <= roi[axis] for axis in range(3)):
            if segmentation_only:
                segmentation_logits = model.segment(image)[0]
            else:
                outputs = model(image, presence)
                segmentation_logits = outputs["segmentation"]
                classification_logits = outputs["classification"]
        else:
            segmentation_logits = sliding_window_inference(
                image,
                roi,
                int(config["inference"].get("sw_batch_size", 1)),
                _segmentation_predictor(model, presence),
                overlap=overlap,
            )
            if not segmentation_only:
                classification_latent = model.encode_for_classification(image)
                classification_logits = model.classify_latent(
                    classification_latent, presence, image.shape[1]
                )
        metrics: dict[str, Any] = {}
        if not classification_only:
            prediction = segmentation_logits.argmax(1).cpu().numpy()[0]
            target = batch["label"].cpu().numpy()[0]
            spacing = tuple(float(value) for value in batch["spacing"][0])
            if official_export_dir is not None:
                export_root = Path(official_export_dir)
                cohort = batch["cohort"][0]
                subject_id = batch["subject_id"][0]
                affine = np.diag((*spacing, 1.0))
                for kind, array in (("prediction", prediction), ("reference", target)):
                    destination = export_root / kind / cohort / f"{subject_id}.nii.gz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), destination)
            evaluation_regions = tuple(
                config["evaluation"].get("regions", ("WT", "TC", "ET", "SNFH"))
            )
            metrics.update(
                segmentation_metrics(
                    prediction,
                    target,
                    spacing,
                    regions=evaluation_regions,
                    minimum_lesion_mm3=float(
                        config["evaluation"].get("component_min_volume_mm3", 100)
                    ),
                    small_lesion_maximum_mm3=float(
                        config["evaluation"].get("small_lesion_max_volume_mm3", 1000)
                    ),
                )
            )
            metrics.update(
                case_volume_metrics(
                    prediction,
                    target,
                    spacing,
                    regions=evaluation_regions,
                    minimum_component_mm3=float(
                        config["evaluation"].get("component_min_volume_mm3", 100)
                    ),
                )
            )
        metrics.update(
            {
                "subject_id": batch["subject_id"][0],
                "cohort": batch["cohort"][0],
                "source_branch": batch["source_branch"][0],
                "source_orientation": batch["source_orientation"][0],
                "has_report": batch["has_report"][0],
                "class_target": int(batch["class_label"][0]),
            }
        )
        if classification_logits is not None:
            probabilities = F.softmax(classification_logits, dim=1).cpu().numpy()[0]
            metrics["class_probability"] = probabilities.tolist()
            metrics["class_predicted"] = int(probabilities.argmax())
            metrics["class_confidence"] = float(probabilities.max())
            metrics["class_correct"] = float(
                int(probabilities.argmax()) == int(batch["class_label"][0])
            )
            all_probabilities.append(probabilities)
            all_labels.append(int(batch["class_label"][0]))
        case_results.append(metrics)
    aggregate = aggregate_case_metrics(case_results)
    aggregate["classification_evaluated"] = bool(all_probabilities)
    if case_results and not classification_only:
        aggregate.update(
            aggregate_volume_metrics(
                case_results,
                regions=tuple(config["evaluation"].get("regions", ("WT", "TC", "ET", "SNFH"))),
            )
        )
        aggregate["stratified"] = stratified_case_metrics(case_results)
    if all_probabilities:
        aggregate.update(
            classification_metrics(
                np.stack(all_probabilities),
                np.asarray(all_labels),
                ece_bins=int(config["evaluation"].get("ece_bins", 15)),
            )
        )
    return aggregate, case_results


def train(
    config: dict[str, Any],
    *,
    device: str | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    seed_everything(int(config["project"]["seed"]))
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    atomic_json_dump(config, output_dir / "resolved_config.json")

    train_dataset = build_dataset(config, "train", True)
    patch_validation = bool(config["training"].get("patch_validation", False))
    validation_dataset = build_dataset(config, "val", False, patch_validation=patch_validation)
    development_limit = development_validation_limit(config)
    validation_dataset.records = stratified_validation_panel(
        validation_dataset.records,
        development_limit,
        int(config["project"]["seed"]),
    )
    atomic_json_dump(
        [
            {
                "subject_id": record["subject_id"],
                "cohort": record["cohort"],
                "source_branch": record["source_branch"],
                "has_report": record.get("has_report", False),
            }
            for record in validation_dataset.records
        ],
        output_dir / "development_validation_panel.json",
    )
    loader_kwargs = {
        "num_workers": int(config["data"].get("num_workers", 0)),
        "collate_fn": collate_training,
        "pin_memory": target_device.type == "cuda",
    }
    sampler = None
    if config["data"].get("class_balanced_sampler", False):
        cohort_counts: dict[str, int] = defaultdict(int)
        for record in train_dataset.records:
            cohort_counts[record["cohort"]] += 1
        weights = [1.0 / cohort_counts[record["cohort"]] for record in train_dataset.records]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    model = build_model(config).to(target_device)
    objective = MultiTaskObjective(
        mode=config["model"].get("joint_weighting", "fixed"),
        lambda_cls=float(config["model"].get("lambda_cls", 0.2)),
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(objective.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["training"]["epochs"]))
    )
    amp_enabled = bool(config["training"].get("amp", True) and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(config["training"].get("accumulation_steps", 1))
    history: list[dict] = []
    best_score = -math.inf
    selection_metric = (
        "balanced_accuracy" if objective.mode == "classification_only" else "macro_dice"
    )
    epochs_without_improvement = 0
    first_epoch = 1
    resume_exact_state_restored = False
    if resume is not None:
        checkpoint = torch.load(resume, map_location=target_device, weights_only=False)
        if checkpoint.get("config_hash") != config_hash(config):
            raise ValueError("Resume checkpoint configuration hash does not match")
        model.load_state_dict(checkpoint["model"])
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch = int(checkpoint["epoch"]) + 1
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        best_score = max(
            (record.get(f"val_{selection_metric}", -math.inf) for record in history),
            default=-math.inf,
        )
        epochs_without_improvement = int(
            checkpoint.get(
                "epochs_without_improvement",
                _epochs_since_best(history, selection_metric),
            )
        )
        resume_exact_state_restored = checkpoint.get("rng_state") is not None
        restore_rng_state(checkpoint.get("rng_state"))
    start_time = time.time()

    for epoch in range(first_epoch, int(config["training"]["epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        objective.train()
        running: dict[str, float] = defaultdict(float)
        epoch_gradients: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        steps = 0
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(target_device, non_blocking=True)
            label = batch["label"].to(target_device, non_blocking=True)
            class_label = batch["class_label"].to(target_device, non_blocking=True)
            presence = batch["modality_presence"].to(target_device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                if objective.mode == "classification_only":
                    latent = model.encode_for_classification(image)
                    outputs = {"classification": model.classify_latent(latent, presence)}
                else:
                    outputs = model(image, presence)
                losses = objective(outputs, label, class_label)
                scaled_loss = losses["loss"] / accumulation
            if step == 1 and objective.mode in {"fixed", "uncertainty", "pcgrad"}:
                epoch_gradients = gradient_diagnostics(
                    model, losses["segmentation_loss"], losses["classification_loss"]
                )
            if objective.mode == "pcgrad":
                pcgrad_result = pcgrad_backward(
                    model,
                    losses["segmentation_loss"],
                    losses["classification_loss"],
                    classification_weight=float(config["model"].get("lambda_cls", 0.2)),
                    accumulation_steps=accumulation,
                    scaler=scaler,
                )
                if step == 1:
                    epoch_gradients.update(pcgrad_result)
            else:
                scaler.scale(scaled_loss).backward()
            if step % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key, value in losses.items():
                running[key] += float(value.detach())
            steps += 1
            maximum = config["training"].get("max_steps_per_epoch")
            if maximum is not None and steps >= int(maximum):
                break
        if steps % accumulation:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        epoch_record = {
            "epoch": epoch,
            **{key: value / max(steps, 1) for key, value in running.items()},
            **epoch_gradients,
        }

        validate_every = int(config["training"].get("validate_every", 1))
        if epoch % validate_every == 0:
            validation, _ = evaluate_model(
                model,
                validation_loader,
                config,
                target_device,
                patch_validation=patch_validation,
                max_cases=config["training"].get("max_validation_cases"),
            )
            epoch_record.update({f"val_{key}": value for key, value in validation.items()})
            score = validation.get(selection_metric, -math.inf)
            if score > best_score:
                best_score = score
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "objective": objective.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "config_hash": config_hash(config),
                        "validation": validation,
                    },
                    output_dir / "checkpoints" / "best.pt",
                )
            else:
                epochs_without_improvement += validate_every
        history.append(epoch_record)
        scheduler.step()
        torch.save(
            {
                "model": model.state_dict(),
                "objective": objective.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "config_hash": config_hash(config),
                "epochs_without_improvement": epochs_without_improvement,
                "rng_state": capture_rng_state(),
            },
            output_dir / "checkpoints" / "last.pt",
        )
        atomic_json_dump(history, output_dir / "history.json")
        if epochs_without_improvement >= int(config["training"].get("patience", 50)):
            break

    summary = {
        "device": str(target_device),
        "epochs_completed": len(history),
        "selection_metric": selection_metric,
        "best_validation_score": best_score,
        f"best_validation_{selection_metric}": best_score,
        "elapsed_seconds": time.time() - start_time,
        "config_hash": config_hash(config),
        "train_subjects": len(train_dataset),
        "validation_subjects": len(validation_dataset),
        "resumed_from": str(resume) if resume else None,
        "resume_exact_state_restored": resume_exact_state_restored,
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def load_checkpoint(model: MultiTaskSegResNet, path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
