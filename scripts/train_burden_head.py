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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tumortrust_vlm.models.burden import FrozenFeatureBurdenMLP
from tumortrust_vlm.utils import (
    atomic_json_dump,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    sha256_file,
    sha256_json,
)

REGIONS = ("WT", "TC", "ET", "SNFH")


def _index_unique(rows: list[dict], source: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        subject_id = row["subject_id"]
        if subject_id in indexed:
            raise ValueError(f"Duplicate subject_id {subject_id!r} in {source}")
        indexed[subject_id] = row
    return indexed


def load_feature_records(features_path: str) -> dict[str, dict]:
    rows = json.loads(Path(features_path).read_text(encoding="utf-8"))
    records = _index_unique(rows, features_path)
    if not records:
        raise ValueError(f"No feature records in {features_path}")
    provenance = {
        json.dumps(row.get("inference_provenance"), sort_keys=True) for row in records.values()
    }
    if any(row.get("inference_provenance") is None for row in records.values()):
        raise ValueError(f"Missing inference_provenance in {features_path}")
    if len(provenance) != 1:
        raise ValueError(f"Inconsistent inference_provenance in {features_path}")
    return records


def feature_provenance(features_path: str) -> dict:
    records = load_feature_records(features_path)
    return next(iter(records.values()))["inference_provenance"]


def assemble(features_path: str, evaluation_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    features = load_feature_records(features_path)
    evaluation_rows = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    evaluation = _index_unique(evaluation_rows, evaluation_path)
    if set(features) != set(evaluation):
        raise ValueError(
            f"Feature/evaluation subject mismatch: {len(features)} versus {len(evaluation)}"
        )
    subject_ids = sorted(features)
    inputs = np.asarray([features[subject_id]["global_features"] for subject_id in subject_ids])
    targets = np.asarray(
        [
            [evaluation[subject_id][f"target_volume_ml_{region}"] for region in REGIONS]
            for subject_id in subject_ids
        ],
        dtype=np.float32,
    )
    if inputs.ndim != 2 or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Burden features/targets must be finite 2D arrays")
    return subject_ids, inputs.astype(np.float32), targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the post-hoc burden MLP on frozen full-volume core features."
    )
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-evaluation", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--val-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--resume")
    args = parser.parse_args()
    seed_everything(args.seed)
    train_subject_ids, train_inputs, train_targets = assemble(
        args.train_features, args.train_evaluation
    )
    val_subject_ids, val_inputs, val_targets = assemble(args.val_features, args.val_evaluation)
    train_feature_provenance = feature_provenance(args.train_features)
    val_feature_provenance = feature_provenance(args.val_features)
    if train_feature_provenance != val_feature_provenance:
        raise ValueError("Train/validation features were not produced by the same frozen core")
    mean = train_inputs.mean(0)
    scale = train_inputs.std(0)
    scale[scale < 1e-6] = 1.0
    train_inputs = (train_inputs - mean) / scale
    val_inputs = (val_inputs - mean) / scale
    loaders = {
        "train": DataLoader(
            TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets)),
            batch_size=args.batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_targets)),
            batch_size=args.batch_size,
            shuffle=False,
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = {
        "input_dim": int(train_inputs.shape[1]),
        "hidden_dim": args.hidden_dim,
        "output_dim": len(REGIONS),
        "dropout": args.dropout,
    }
    training_configuration = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    data_provenance = {
        "train_features_sha256": sha256_file(args.train_features),
        "train_evaluation_sha256": sha256_file(args.train_evaluation),
        "validation_features_sha256": sha256_file(args.val_features),
        "validation_evaluation_sha256": sha256_file(args.val_evaluation),
        "train_subjects_sha256": sha256_json(train_subject_ids),
        "validation_subjects_sha256": sha256_json(val_subject_ids),
    }
    model = FrozenFeatureBurdenMLP(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    first_epoch = 1
    best = float("inf")
    epochs_without_improvement = 0
    resume_exact_state_restored = False
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["architecture"] != architecture:
            raise ValueError("Burden resume architecture does not match")
        if checkpoint.get("training_configuration") != training_configuration:
            raise ValueError("Burden resume training configuration does not match")
        if checkpoint.get("data_provenance") != data_provenance:
            raise ValueError("Burden resume inputs do not match")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", [])
        best = float(checkpoint.get("best_validation_loss", float("inf")))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        resume_exact_state_restored = checkpoint.get("rng_state") is not None
        restore_rng_state(checkpoint.get("rng_state"))

    def payload(epoch: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": architecture,
            "training_configuration": training_configuration,
            "data_provenance": data_provenance,
            "core_feature_provenance": train_feature_provenance,
            "normalization_mean": mean,
            "normalization_scale": scale,
            "regions": REGIONS,
            "epoch": epoch,
            "history": history,
            "best_validation_loss": best,
            "epochs_without_improvement": epochs_without_improvement,
            "rng_state": capture_rng_state(),
        }

    for epoch in range(first_epoch, args.epochs + 1):
        record = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train")
            total = 0.0
            subjects = 0
            for inputs, targets in loaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model(inputs)
                    loss = F.smooth_l1_loss(prediction, torch.log1p(targets))
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach()) * len(inputs)
                subjects += len(inputs)
            record[f"{phase}_loss"] = total / subjects
        history.append(record)
        if record["val_loss"] < best:
            best = record["val_loss"]
            epochs_without_improvement = 0
            torch.save(payload(epoch), output / "best.pt")
        else:
            epochs_without_improvement += 1
        torch.save(payload(epoch), output / "last.pt")
        atomic_json_dump(history, output / "history.json")
        if epochs_without_improvement >= args.patience:
            break
    summary = {
        "train_subjects": len(train_inputs),
        "validation_subjects": len(val_inputs),
        "epochs_completed": len(history),
        "best_validation_log_volume_loss": best,
        "frozen_full_volume_features": True,
        "mask_derived_volume_used_as_input": False,
        "data_provenance": data_provenance,
        "core_feature_provenance": train_feature_provenance,
        "resume_exact_state_restored": resume_exact_state_restored,
    }
    atomic_json_dump(summary, output / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
