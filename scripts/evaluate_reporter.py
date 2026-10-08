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

from tumortrust_vlm.evaluation.reporting import (
    bertscore,
    bleu,
    evidence_consistency,
    report_diversity,
    rouge_l,
)
from tumortrust_vlm.models.reporter import EvidenceConditionedReporter
from tumortrust_vlm.reporting.dataset import ReporterDataset, WordTokenizer, collate_reporter
from tumortrust_vlm.reporting.evidence import EvidenceCard
from tumortrust_vlm.reporting.findings import (
    extract_normalized_findings,
    structured_finding_metrics,
    structured_laterality_metrics,
)
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("text_only", "evidence_only", "visual_only", "full"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-length", type=int)
    parser.add_argument("--schema-constrained", action="store_true")
    parser.add_argument("--bertscore-model")
    parser.add_argument("--bertscore-device")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()

    raw_records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if (
        any(record.get("report_split") == "test" for record in raw_records)
        and not args.unlock_final_test
    ):
        raise SystemExit("Final report test is locked")
    by_subject = {record["subject_id"]: record for record in raw_records}
    if len(by_subject) != len(raw_records):
        raise ValueError("Reporter evaluation records contain duplicate subject IDs")
    checkpoint_path = Path(args.checkpoint)
    tokenizer = WordTokenizer.load(checkpoint_path.parent / "vocabulary.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_configuration = checkpoint.get("training_configuration", {})
    maximum_length = args.maximum_length or int(
        training_configuration.get("maximum_length", 256)
    )
    if args.maximum_length and training_configuration.get("maximum_length") not in {
        None,
        args.maximum_length,
    }:
        raise ValueError("Evaluation maximum length does not match reporter training")
    data_provenance = checkpoint.get("data_provenance", {})
    vocabulary_hash = data_provenance.get("vocabulary_sha256")
    if vocabulary_hash and vocabulary_hash != sha256_file(checkpoint_path.parent / "vocabulary.json"):
        raise ValueError("Reporter vocabulary does not match checkpoint provenance")
    record_splits = {record.get("report_split") for record in raw_records}
    if len(record_splits) != 1:
        raise ValueError("Reporter evaluation records mix report splits")
    record_split = record_splits.pop()
    expected_input_hash = data_provenance.get(
        "train_records_sha256" if record_split == "train" else "validation_records_sha256"
    )
    if (
        record_split in {"train", "val"}
        and expected_input_hash
        and expected_input_hash != sha256_file(args.records)
    ):
        raise ValueError("Reporter development records do not match checkpoint provenance")
    dataset = ReporterDataset(args.records, tokenizer, maximum_length)
    first = dataset[0]
    visual_dim = first["visual_tokens"].shape[-1] if first["visual_tokens"] is not None else 1
    architecture = checkpoint.get(
        "architecture",
        {
            "vocab_size": len(tokenizer),
            "visual_dim": visual_dim,
            "evidence_dim": len(first["evidence"]),
            "hidden_dim": 512,
            "embedding_dim": 256,
        },
    )
    model = EvidenceConditionedReporter(**architecture).to(device)
    if checkpoint.get("mode") != args.mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint.get('mode')} does not match requested {args.mode}"
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reporter,
    )
    predictions = []
    with torch.no_grad():
        for batch in loader:
            evidence = (
                batch["evidence"].to(device) if args.mode in ("evidence_only", "full") else None
            )
            visual = batch["visual_tokens"]
            visual = (
                visual.to(device)
                if visual is not None and args.mode in ("visual_only", "full")
                else None
            )
            generated = model.generate(
                tokenizer.bos_id,
                tokenizer.eos_id,
                maximum_length,
                visual_tokens=visual,
                evidence=evidence,
                batch_size=len(batch["subject_id"]),
            ).cpu()
            for subject_id, token_ids in zip(batch["subject_id"], generated.tolist(), strict=True):
                record = by_subject[subject_id]
                learned_text = tokenizer.decode(token_ids)
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
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mode": args.mode,
        "schema_constrained": args.schema_constrained,
        "maximum_length": maximum_length,
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


if __name__ == "__main__":
    main()
