#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tumortrust_vlm.models.reporter import EvidenceConditionedReporter
from tumortrust_vlm.reporting.dataset import (
    ReporterDataset,
    WordTokenizer,
    collate_reporter,
    validate_reporter_record_sets,
)
from tumortrust_vlm.utils import (
    atomic_json_dump,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--val-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("text_only", "evidence_only", "visual_only", "full"), default="full"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--lambda-field", type=float, default=0.20)
    parser.add_argument("--lambda-consistency", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--minimum-token-frequency", type=int, default=2)
    parser.add_argument("--maximum-vocabulary-size", type=int, default=16000)
    parser.add_argument(
        "--vocabulary",
        help="Frozen WordTokenizer vocabulary. Required by the locked T10 recipe.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_records = json.loads(Path(args.train_records).read_text(encoding="utf-8"))
    val_records = json.loads(Path(args.val_records).read_text(encoding="utf-8"))
    record_audit = validate_reporter_record_sets(train_records, val_records)
    vocabulary_path = output / "vocabulary.json"
    if args.resume:
        tokenizer = WordTokenizer.load(vocabulary_path)
    elif args.vocabulary:
        tokenizer = WordTokenizer.load(args.vocabulary)
        tokenizer.save(vocabulary_path)
    else:
        tokenizer = WordTokenizer.fit(
            [record["report_text"] for record in train_records],
            minimum_frequency=args.minimum_token_frequency,
            maximum_size=args.maximum_vocabulary_size,
        )
        tokenizer.save(vocabulary_path)
    train_dataset = ReporterDataset(args.train_records, tokenizer, args.maximum_length)
    val_dataset = ReporterDataset(args.val_records, tokenizer, args.maximum_length)
    first = train_dataset[0]
    visual_dim = first["visual_tokens"].shape[-1] if first["visual_tokens"] is not None else 1
    evidence_dim = len(first["evidence"])
    architecture = {
        "vocab_size": len(tokenizer),
        "visual_dim": visual_dim,
        "evidence_dim": evidence_dim,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
    }
    model = EvidenceConditionedReporter(**architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_reporter
        ),
        "val": DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_reporter
        ),
    }
    best = float("inf")
    history = []
    first_epoch = 1
    resume_checkpoint = None
    training_configuration = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lambda_field": args.lambda_field,
        "lambda_consistency": args.lambda_consistency,
        "maximum_length": args.maximum_length,
        "minimum_token_frequency": args.minimum_token_frequency,
        "maximum_vocabulary_size": args.maximum_vocabulary_size,
        "learning_rate": args.learning_rate,
        "vocabulary_source_sha256": (
            sha256_file(args.vocabulary) if args.vocabulary else None
        ),
    }
    data_provenance = {
        "train_records_sha256": sha256_file(args.train_records),
        "validation_records_sha256": sha256_file(args.val_records),
        "vocabulary_sha256": sha256_file(vocabulary_path),
        **record_audit,
    }
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            resume_checkpoint["architecture"] != architecture
            or resume_checkpoint["mode"] != args.mode
        ):
            raise ValueError("Resume checkpoint architecture/mode does not match")
        if resume_checkpoint.get("training_configuration") != training_configuration:
            raise ValueError("Resume reporter training configuration does not match")
        if resume_checkpoint.get("data_provenance") != data_provenance:
            raise ValueError("Resume reporter inputs do not match")
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        restore_rng_state(resume_checkpoint.get("rng_state"))
        first_epoch = int(resume_checkpoint["epoch"]) + 1
        history = resume_checkpoint.get("history", [])
        best = float(resume_checkpoint.get("best_validation_loss", float("inf")))

    def checkpoint_payload(epoch: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mode": args.mode,
            "epoch": epoch,
            "architecture": architecture,
            "lambda_field": args.lambda_field,
            "lambda_consistency": args.lambda_consistency,
            "seed": args.seed,
            "training_configuration": training_configuration,
            "data_provenance": data_provenance,
            "history": history,
            "best_validation_loss": best,
            "rng_state": capture_rng_state(),
        }

    for epoch in range(first_epoch, args.epochs + 1):
        epoch_result = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train")
            totals = {"loss": 0.0, "text_loss": 0.0, "field_loss": 0.0, "consistency_loss": 0.0}
            count = 0
            for batch in loaders[phase]:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                evidence = (
                    batch["evidence"].to(device) if args.mode in ("evidence_only", "full") else None
                )
                visual = batch["visual_tokens"]
                visual = (
                    visual.to(device)
                    if visual is not None and args.mode in ("visual_only", "full")
                    else None
                )
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(input_ids, visual_tokens=visual, evidence=evidence)
                    text_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
                    field_loss = logits.new_zeros(())
                    if visual is not None or evidence is not None:
                        reconstructed = model.reconstruct_fields(
                            visual, evidence, input_ids.shape[0]
                        )
                        field_loss = F.smooth_l1_loss(reconstructed, batch["evidence"].to(device))
                    consistency_loss = logits.new_zeros(())
                    if visual is not None and evidence is not None:
                        consistency_loss = model.source_consistency_loss(visual, evidence)
                    loss = (
                        text_loss
                        + args.lambda_field * field_loss
                        + args.lambda_consistency * consistency_loss
                    )
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                for key, value in (
                    ("loss", loss),
                    ("text_loss", text_loss),
                    ("field_loss", field_loss),
                    ("consistency_loss", consistency_loss),
                ):
                    totals[key] += float(value.detach())
                count += 1
            for key, value in totals.items():
                epoch_result[f"{phase}_{key}"] = value / max(count, 1)
        history.append(epoch_result)
        if epoch_result["val_loss"] < best:
            best = epoch_result["val_loss"]
            torch.save(checkpoint_payload(epoch), output / "best.pt")
        torch.save(checkpoint_payload(epoch), output / "last.pt")
        atomic_json_dump(history, output / "history.json")
    summary = {
        "mode": args.mode,
        "best_validation_loss": best,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "vocabulary": len(tokenizer),
        "lambda_field": args.lambda_field,
        "lambda_consistency": args.lambda_consistency,
        "resume_exact_state_restored": bool(
            args.resume and resume_checkpoint.get("rng_state") is not None
        ),
        "data_provenance": data_provenance,
    }
    atomic_json_dump(summary, output / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
