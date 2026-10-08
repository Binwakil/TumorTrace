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
from transformers import AutoTokenizer

from tumortrust_vlm.evaluation.reporting import (
    extract_evidence_fields,
    targeted_intervention_accuracy,
)
from tumortrust_vlm.models.vicuna_reporter import PrefixConditionedCausalReporter
from tumortrust_vlm.reporting.dataset import ReporterDataset, collate_reporter
from tumortrust_vlm.reporting.evidence import EvidenceCard
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.reporting.vicuna_dataset import encode_prompt
from tumortrust_vlm.utils import atomic_json_dump, model_artifact_provenance

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


class _UnusedTokenizer:
    def encode(self, text: str, maximum_length: int = 256) -> list[int]:
        return [1, 2]


def changed_fraction(before: list[str], after: list[str]) -> float:
    return float(np.mean([left != right for left, right in zip(before, after, strict=True)]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe evidence and 3D-token use in a frozen-backbone 7B prefix reporter."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("7B prefix reporter probes require CUDA")

    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if any(record.get("report_split") == "test" for record in records) and not args.unlock_final_test:
        raise SystemExit("Final report test is locked")
    by_subject = {record["subject_id"]: record for record in records}
    if len(by_subject) != len(records):
        raise ValueError("Reporter probe records contain duplicate subject IDs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if checkpoint.get("base_model_provenance") != model_artifact_provenance(
        architecture["model_path"]
    ):
        raise ValueError("Reporter probe base-model files do not match checkpoint")
    mode = checkpoint["mode"]
    tokenizer = AutoTokenizer.from_pretrained(architecture["model_path"], use_fast=False)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
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

    def decode(
        visual: torch.Tensor | None,
        evidence: torch.Tensor | None,
        batch_size: int,
    ) -> list[str]:
        input_ids, attention_mask = encode_prompt(tokenizer, batch_size)
        generated = model.generate(
            input_ids.to(device),
            attention_mask.to(device),
            visual_tokens=visual,
            evidence=evidence,
            maximum_new_tokens=args.maximum_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        ).cpu()
        return [
            tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            for token_ids in generated.tolist()
        ]

    targeted = {
        field: {"before": [], "after": [], "expected": []}
        for field in EVIDENCE_GROUPS
    }
    visual_probe_rows = []
    with torch.no_grad():
        for batch in loader:
            size = len(batch["subject_id"])
            evidence = (
                batch["evidence"].to(device) if mode in ("evidence_only", "full") else None
            )
            visual = (
                batch["visual_tokens"].to(device) if mode in ("visual_only", "full") else None
            )
            base_text = decode(visual, evidence, size)
            base_fields = [extract_evidence_fields(text) for text in base_text]
            if evidence is not None and evidence.shape[1] >= 16 and size > 1:
                donor_subjects = batch["subject_id"][-1:] + batch["subject_id"][:-1]
                donor_expected = [
                    extract_evidence_fields(
                        render_findings(EvidenceCard.from_dict(by_subject[subject_id]["evidence"]))
                    )
                    for subject_id in donor_subjects
                ]
                rolled = torch.roll(evidence, shifts=1, dims=0)
                field_items = list(EVIDENCE_GROUPS.items())
                for start in range(0, len(field_items), 2):
                    chunk = field_items[start : start + 2]
                    changed_batches = []
                    for _, location in chunk:
                        changed = evidence.clone()
                        changed[:, location] = rolled[:, location]
                        changed_batches.append(changed)
                    repeated_visual = (
                        torch.cat([visual] * len(chunk), dim=0)
                        if visual is not None
                        else None
                    )
                    changed_text = decode(
                        repeated_visual,
                        torch.cat(changed_batches, dim=0),
                        size * len(chunk),
                    )
                    for chunk_index, (field, _) in enumerate(chunk):
                        selected = changed_text[
                            chunk_index * size : (chunk_index + 1) * size
                        ]
                        targeted[field]["before"].extend(base_fields)
                        targeted[field]["after"].extend(
                            extract_evidence_fields(text) for text in selected
                        )
                        targeted[field]["expected"].extend(donor_expected)
            if visual is not None and size > 1:
                visual_variants = (
                    torch.roll(visual, shifts=1, dims=0),
                    torch.zeros_like(visual),
                    visual * 1.01,
                )
                combined = decode(
                    torch.cat(visual_variants, dim=0),
                    torch.cat([evidence] * 3, dim=0) if evidence is not None else None,
                    size * 3,
                )
                swapped = combined[:size]
                zeroed = combined[size : 2 * size]
                perturbed = combined[2 * size :]
                for index, subject_id in enumerate(batch["subject_id"]):
                    visual_probe_rows.append(
                        {
                            "subject_id": subject_id,
                            "base": base_text[index],
                            "image_swap": swapped[index],
                            "zero_image": zeroed[index],
                            "light_perturbation": perturbed[index],
                        }
                    )

    targeted_scores = {}
    for field, values in targeted.items():
        score = targeted_intervention_accuracy(
            values["before"], values["after"], values["expected"], field
        )
        targeted_scores[field] = None if math.isnan(score) else score
    available = [score for score in targeted_scores.values() if score is not None]
    summary = {
        "mode": mode,
        "subjects": len(dataset),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "targeted_evidence_intervention_accuracy": (
            float(np.mean(available)) if available else None
        ),
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
