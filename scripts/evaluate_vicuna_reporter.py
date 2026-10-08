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
from transformers import AutoTokenizer

from tumortrust_vlm.evaluation.reporting import (
    bertscore,
    bleu,
    evidence_consistency,
    report_diversity,
    rouge_l,
)
from tumortrust_vlm.models.vicuna_reporter import PrefixConditionedCausalReporter
from tumortrust_vlm.reporting.dataset import ReporterDataset, collate_reporter
from tumortrust_vlm.reporting.evidence import EvidenceCard
from tumortrust_vlm.reporting.findings import (
    extract_normalized_findings,
    structured_finding_metrics,
    structured_laterality_metrics,
)
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.reporting.vicuna_dataset import encode_prompt
from tumortrust_vlm.utils import (
    atomic_json_dump,
    model_artifact_provenance,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--schema-constrained", action="store_true")
    parser.add_argument("--bertscore-model")
    parser.add_argument("--bertscore-device")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("7B prefix reporter evaluation requires CUDA")
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if (
        any(record.get("report_split") == "test" for record in records)
        and not args.unlock_final_test
    ):
        raise SystemExit("Final report test is locked")
    by_subject = {record["subject_id"]: record for record in records}
    if len(by_subject) != len(records):
        raise ValueError("Vicuna evaluation records contain duplicate subject IDs")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if checkpoint.get("base_model_provenance") != model_artifact_provenance(
        architecture["model_path"]
    ):
        raise ValueError("Vicuna evaluation base-model files do not match checkpoint")
    record_splits = {record.get("report_split") for record in records}
    if len(record_splits) != 1:
        raise ValueError("Vicuna evaluation records mix report splits")
    record_split = record_splits.pop()
    data_provenance = checkpoint.get("data_provenance", {})
    expected_input_hash = data_provenance.get(
        "train_records_sha256" if record_split == "train" else "validation_records_sha256"
    )
    if (
        record_split in {"train", "val"}
        and expected_input_hash
        and expected_input_hash != sha256_file(args.records)
    ):
        raise ValueError("Vicuna development records do not match checkpoint provenance")
    tokenizer = AutoTokenizer.from_pretrained(architecture["model_path"], use_fast=False)
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    dataset = ReporterDataset(args.records, _UnusedTokenizer())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reporter,
    )
    device = torch.device("cuda")
    model = PrefixConditionedCausalReporter(**architecture).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_state"])
    model.eval()
    mode = checkpoint["mode"]
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch["subject_id"])
            input_ids, attention_mask = encode_prompt(tokenizer, batch_size)
            visual = batch["visual_tokens"].to(device) if mode in ("visual_only", "full") else None
            evidence = batch["evidence"].to(device) if mode in ("evidence_only", "full") else None
            generated = model.generate(
                input_ids.to(device),
                attention_mask.to(device),
                visual_tokens=visual,
                evidence=evidence,
                maximum_new_tokens=args.maximum_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            ).cpu()
            for subject_id, token_ids in zip(batch["subject_id"], generated.tolist(), strict=True):
                record = by_subject[subject_id]
                learned_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                text = learned_text
                if args.schema_constrained:
                    structured = render_findings(EvidenceCard.from_dict(record["evidence"]))
                    text = f"{structured} Narrative: {learned_text}" if learned_text else structured
                consistency = evidence_consistency(text, record["evidence"])
                reference_findings = record.get("normalized_findings") or extract_normalized_findings(
                    record["report_text"]
                )
                predictions.append(
                    {
                        "subject_id": subject_id,
                        "report_split": record.get("report_split"),
                        "reference": record["report_text"],
                        "prediction": text,
                        "bleu_1": bleu(text, record["report_text"], 1),
                        "bleu_4": bleu(text, record["report_text"], 4),
                        "rouge_l": rouge_l(text, record["report_text"]),
                        "predicted_normalized_findings": extract_normalized_findings(text),
                        "reference_normalized_findings": reference_findings,
                        **consistency,
                    }
                )
    texts = [record["prediction"] for record in predictions]
    bertscore_result = None
    if args.bertscore_model:
        bertscore_result = bertscore(
            texts,
            [record["reference"] for record in predictions],
            model_type=args.bertscore_model,
            device=args.bertscore_device,
        )
        for index, record in enumerate(predictions):
            record["bertscore_precision"] = bertscore_result["precision"][index]
            record["bertscore_recall"] = bertscore_result["recall"][index]
            record["bertscore_f1"] = bertscore_result["f1"][index]
    summary = {
        "subjects": len(predictions),
        "mode": mode,
        "checkpoint_epoch": checkpoint["epoch"],
        "schema_constrained": args.schema_constrained,
        "bleu_policy": "clipped_add_one_smoothed_effective_order_v1",
        "bleu_1": float(np.mean([record["bleu_1"] for record in predictions])),
        "bleu_4": float(np.mean([record["bleu_4"] for record in predictions])),
        "rouge_l": float(np.mean([record["rouge_l"] for record in predictions])),
        "bertscore_status": "computed" if bertscore_result else "not_requested",
        "bertscore_model": bertscore_result["model_type"] if bertscore_result else None,
        "bertscore_package_version": (
            bertscore_result["package_version"] if bertscore_result else None
        ),
        "bertscore_precision": (
            float(np.mean(bertscore_result["precision"])) if bertscore_result else None
        ),
        "bertscore_recall": (
            float(np.mean(bertscore_result["recall"])) if bertscore_result else None
        ),
        "bertscore_f1": (
            float(np.mean(bertscore_result["f1"])) if bertscore_result else None
        ),
        "structured_field_recall": float(
            np.mean([record["structured_field_recall"] for record in predictions])
        ),
        "structured_field_unsupported_rate": float(
            np.mean([record["structured_field_unsupported_rate"] for record in predictions])
        ),
        "structured_field_contradiction_rate": float(
            np.mean([record["structured_field_contradiction_rate"] for record in predictions])
        ),
        "broad_unsupported_claim_rate": None,
        "broad_unsupported_claim_status": (
            "requires_versioned_clinical_parser_or_radiologist_review"
        ),
        **report_diversity(texts),
        **structured_finding_metrics(
            texts,
            [record["reference_normalized_findings"] for record in predictions],
        ),
        **structured_laterality_metrics(
            texts,
            [record["reference_normalized_findings"] for record in predictions],
        ),
    }
    output = Path(args.output)
    atomic_json_dump(predictions, output)
    atomic_json_dump(summary, output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


class _UnusedTokenizer:
    """ReporterDataset adapter; only evidence/visual tensors are used here."""

    def encode(self, text: str, maximum_length: int = 256) -> list[int]:
        return [1, 2]


if __name__ == "__main__":
    main()
