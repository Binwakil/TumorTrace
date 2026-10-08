#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tumortrust_vlm.evaluation.reporting import (
    extract_evidence_fields,
    targeted_intervention_accuracy,
)
from tumortrust_vlm.models.reporter import EvidenceConditionedReporter
from tumortrust_vlm.reporting.dataset import ReporterDataset, WordTokenizer, collate_reporter
from tumortrust_vlm.reporting.evidence import EvidenceCard
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.utils import atomic_json_dump

EVIDENCE_GROUPS: dict[str, slice | int] = {
    "family": slice(0, 3),
    "volume_WT": 3,
    "volume_TC": 4,
    "volume_ET": 5,
    "volume_SNFH": 6,
    "laterality": slice(7, 12),
    "component_count": 12,
    "referral": 15,
}


def decode_batch(
    model: EvidenceConditionedReporter,
    tokenizer: WordTokenizer,
    maximum_length: int,
    *,
    visual: torch.Tensor | None,
    evidence: torch.Tensor | None,
    batch_size: int,
) -> list[str]:
    generated = model.generate(
        tokenizer.bos_id,
        tokenizer.eos_id,
        maximum_length,
        visual_tokens=visual,
        evidence=evidence,
        batch_size=batch_size,
    ).cpu()
    return [tokenizer.decode(ids) for ids in generated.tolist()]


def changed_fraction(before: list[str], after: list[str]) -> float:
    return float(np.mean([left != right for left, right in zip(before, after, strict=True)]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evidence/image swap, zero-image, and light-perturbation reporter probes."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("text_only", "evidence_only", "visual_only", "full"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()

    raw_records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if (
        any(record.get("report_split") == "test" for record in raw_records)
        and not args.unlock_final_test
    ):
        raise SystemExit("Final report test is locked")
    by_subject = {record["subject_id"]: record for record in raw_records}
    checkpoint_path = Path(args.checkpoint)
    tokenizer = WordTokenizer.load(checkpoint_path.parent / "vocabulary.json")
    dataset = ReporterDataset(args.records, tokenizer, args.maximum_length)
    first = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get(
        "architecture",
        {
            "vocab_size": len(tokenizer),
            "visual_dim": first["visual_tokens"].shape[-1]
            if first["visual_tokens"] is not None
            else 1,
            "evidence_dim": len(first["evidence"]),
            "hidden_dim": 512,
            "embedding_dim": 256,
        },
    )
    if checkpoint.get("mode") != args.mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint.get('mode')} does not match requested {args.mode}"
        )
    model = EvidenceConditionedReporter(**architecture).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reporter,
        drop_last=False,
    )

    targeted: dict[str, dict[str, list[dict]]] = {
        field: {"before": [], "after": [], "expected": []}
        for field in EVIDENCE_GROUPS
    }
    visual_probe_rows = []
    with torch.no_grad():
        for batch in loader:
            size = len(batch["subject_id"])
            evidence = (
                batch["evidence"].to(device)
                if args.mode in ("evidence_only", "full")
                else None
            )
            visual = batch["visual_tokens"]
            visual = (
                visual.to(device)
                if visual is not None and args.mode in ("visual_only", "full")
                else None
            )
            base_text = decode_batch(
                model,
                tokenizer,
                args.maximum_length,
                visual=visual,
                evidence=evidence,
                batch_size=size,
            )
            base_fields = [extract_evidence_fields(text) for text in base_text]

            if evidence is not None and evidence.shape[1] >= 16 and size > 1:
                donor_subjects = batch["subject_id"][-1:] + batch["subject_id"][:-1]
                donor_expected = [
                    extract_evidence_fields(
                        render_findings(
                            EvidenceCard.from_dict(by_subject[subject_id]["evidence"])
                        )
                    )
                    for subject_id in donor_subjects
                ]
                rolled = torch.roll(evidence, shifts=1, dims=0)
                for field, location in EVIDENCE_GROUPS.items():
                    changed = evidence.clone()
                    changed[:, location] = rolled[:, location]
                    changed_text = decode_batch(
                        model,
                        tokenizer,
                        args.maximum_length,
                        visual=visual,
                        evidence=changed,
                        batch_size=size,
                    )
                    targeted[field]["before"].extend(base_fields)
                    targeted[field]["after"].extend(
                        extract_evidence_fields(text) for text in changed_text
                    )
                    targeted[field]["expected"].extend(donor_expected)

            if visual is not None and size > 1:
                swapped_text = decode_batch(
                    model,
                    tokenizer,
                    args.maximum_length,
                    visual=torch.roll(visual, shifts=1, dims=0),
                    evidence=evidence,
                    batch_size=size,
                )
                zero_text = decode_batch(
                    model,
                    tokenizer,
                    args.maximum_length,
                    visual=torch.zeros_like(visual),
                    evidence=evidence,
                    batch_size=size,
                )
                perturbed_text = decode_batch(
                    model,
                    tokenizer,
                    args.maximum_length,
                    visual=visual * 1.01,
                    evidence=evidence,
                    batch_size=size,
                )
                for index, subject_id in enumerate(batch["subject_id"]):
                    visual_probe_rows.append(
                        {
                            "subject_id": subject_id,
                            "base": base_text[index],
                            "image_swap": swapped_text[index],
                            "zero_image": zero_text[index],
                            "light_perturbation": perturbed_text[index],
                        }
                    )

    targeted_scores = {}
    for field, values in targeted.items():
        score = targeted_intervention_accuracy(
            values["before"], values["after"], values["expected"], field
        )
        targeted_scores[field] = None if math.isnan(score) else score
    available_scores = [score for score in targeted_scores.values() if score is not None]
    summary = {
        "mode": args.mode,
        "subjects": len(dataset),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "targeted_evidence_intervention_accuracy": float(np.mean(available_scores))
        if available_scores
        else None,
        "targeted_evidence_intervention_by_field": targeted_scores,
    }
    if visual_probe_rows:
        base = [row["base"] for row in visual_probe_rows]
        for key in ("image_swap", "zero_image", "light_perturbation"):
            summary[f"{key}_output_change_fraction"] = changed_fraction(
                base, [row[key] for row in visual_probe_rows]
            )

    output = Path(args.output)
    atomic_json_dump(visual_probe_rows, output)
    atomic_json_dump(summary, output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
