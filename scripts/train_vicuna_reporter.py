#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tumortrust_vlm.models.vicuna_reporter import (
    PrefixConditionedCausalReporter,
    local_decoder_family,
)
from tumortrust_vlm.reporting.dataset import validate_reporter_record_sets
from tumortrust_vlm.reporting.vicuna_dataset import (
    VicunaReporterDataset,
    collate_vicuna,
)
from tumortrust_vlm.utils import (
    atomic_json_dump,
    capture_rng_state,
    model_artifact_provenance,
    restore_rng_state,
    seed_everything,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen-backbone LLaVA language decoder with LoRA prefix adapters."
    )
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--val-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="../shared/models/llava-v1.5-7b")
    parser.add_argument(
        "--mode",
        choices=("text_only", "evidence_only", "visual_only", "full"),
        default="full",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--protocol", default="configs/vlm_decoder_comparison.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Vicuna reporter training requires CUDA")
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_records = json.loads(Path(args.train_records).read_text(encoding="utf-8"))
    val_records = json.loads(Path(args.val_records).read_text(encoding="utf-8"))
    record_audit = validate_reporter_record_sets(train_records, val_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    train_dataset = VicunaReporterDataset(args.train_records, tokenizer, args.maximum_length)
    val_dataset = VicunaReporterDataset(args.val_records, tokenizer, args.maximum_length)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Reporter train and validation records must both be non-empty")
    first = train_dataset[0]
    architecture = {
        "model_path": args.model,
        "decoder_family": local_decoder_family(args.model),
        "visual_dim": int(first["visual_tokens"].shape[-1]),
        "evidence_dim": int(first["evidence"].numel()),
        "prefix_tokens": 8,
        "context_dim": 512,
        "lora_rank": 8,
        "lora_alpha": 16,
    }
    device = torch.device("cuda")
    model = PrefixConditionedCausalReporter(**architecture).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    collate = partial(collate_vicuna, pad_token_id=pad_token_id)
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
        ),
    }
    best = float("inf")
    history = []
    first_epoch = 1
    resume_checkpoint = None
    training_configuration = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "maximum_length": args.maximum_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "protocol_path": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
    }
    base_model_provenance = model_artifact_provenance(args.model)
    data_provenance = {
        "train_records_sha256": sha256_file(args.train_records),
        "validation_records_sha256": sha256_file(args.val_records),
        **record_audit,
    }
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            resume_checkpoint["architecture"] != architecture
            or resume_checkpoint["mode"] != args.mode
        ):
            raise ValueError("Resume checkpoint architecture/mode does not match")
        if resume_checkpoint.get("training_configuration") != training_configuration:
            raise ValueError("Resume Vicuna training configuration does not match")
        if resume_checkpoint.get("data_provenance") != data_provenance:
            raise ValueError("Resume Vicuna reporter inputs do not match")
        if resume_checkpoint.get("base_model_provenance") != base_model_provenance:
            raise ValueError("Resume Vicuna base-model files do not match")
        model.load_trainable_state_dict(resume_checkpoint["trainable_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        restore_rng_state(resume_checkpoint.get("rng_state"))
        first_epoch = int(resume_checkpoint["epoch"]) + 1
        history = resume_checkpoint.get("history", [])
        best = float(resume_checkpoint.get("best_validation_loss", float("inf")))

    def checkpoint_payload(epoch: int) -> dict:
        return {
            "trainable_state": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": architecture,
            "mode": args.mode,
            "epoch": epoch,
            "seed": args.seed,
            "maximum_length": args.maximum_length,
            "training_configuration": training_configuration,
            "data_provenance": data_provenance,
            "base_model_provenance": base_model_provenance,
            "history": history,
            "best_validation_loss": best,
            "rng_state": capture_rng_state(),
        }

    for epoch in range(first_epoch, args.epochs + 1):
        epoch_record = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train")
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            batches = 0
            for step, batch in enumerate(loaders[phase], start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                visual = (
                    batch["visual_tokens"].to(device)
                    if args.mode in ("visual_only", "full")
                    else None
                )
                evidence = (
                    batch["evidence"].to(device) if args.mode in ("evidence_only", "full") else None
                )
                with torch.set_grad_enabled(phase == "train"):
                    result = model(
                        input_ids,
                        attention_mask,
                        labels,
                        visual,
                        evidence,
                    )
                    loss = result.loss
                    if phase == "train":
                        (loss / args.accumulation_steps).backward()
                        if step % args.accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach())
                batches += 1
            if phase == "train" and batches % args.accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_record[f"{phase}_loss"] = total_loss / max(1, batches)
        history.append(epoch_record)
        if epoch_record["val_loss"] < best:
            best = epoch_record["val_loss"]
            torch.save(checkpoint_payload(epoch), output / "best.pt")
        torch.save(checkpoint_payload(epoch), output / "last.pt")
        atomic_json_dump(history, output / "history.json")
    summary = {
        "mode": args.mode,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_validation_loss": best,
        "train_subjects": len(train_dataset),
        "validation_subjects": len(val_dataset),
        **model.trainable_parameter_summary(),
        "clip_loaded": False,
        "inherited_llava_projector_loaded": False,
        "resume_exact_state_restored": bool(
            args.resume and resume_checkpoint.get("rng_state") is not None
        ),
        "data_provenance": data_provenance,
        "base_model_files_hashed": len(base_model_provenance),
        "decoder_family": architecture["decoder_family"],
    }
    atomic_json_dump(summary, output / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
