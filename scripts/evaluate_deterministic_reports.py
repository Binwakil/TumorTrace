#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.reporting import (
    bertscore,
    bleu,
    evidence_consistency,
    extract_evidence_fields,
    intervention_accuracy,
    report_diversity,
    rouge_l,
)
from tumortrust_vlm.reporting.evidence import EvidenceCard, VolumeEvidence
from tumortrust_vlm.reporting.findings import (
    extract_normalized_findings,
    structured_finding_metrics,
)
from tumortrust_vlm.reporting.renderer import render_findings, validate_rendered_values
from tumortrust_vlm.utils import atomic_json_dump


def intervention_cards(card: EvidenceCard) -> dict[str, EvidenceCard]:
    """Create one-field counterfactual cards for deterministic grounding checks."""
    unavailable = set(card.unavailable_fields)
    interventions: dict[str, EvidenceCard] = {}
    if "tumor_family" not in unavailable:
        families = tuple(card.tumor_family_probabilities)
        current = families.index(card.predicted_family)
        replacement = families[(current + 1) % len(families)]
        probabilities = {family: 0.0 for family in families}
        probabilities[replacement] = 1.0
        interventions["family"] = replace(
            card,
            tumor_family_probabilities=probabilities,
            predicted_family=replacement,
        )
    for region, evidence in card.volumes.items():
        field = f"volume_{region}"
        if "volume" in unavailable or field in unavailable:
            continue
        increment = max(1.0, abs(evidence.value_ml) * 0.10)
        volumes = dict(card.volumes)
        volumes[region] = VolumeEvidence(
            value_ml=evidence.value_ml + increment,
            interval_90_ml=evidence.interval_90_ml,
            interval_95_ml=evidence.interval_95_ml,
        )
        interventions[field] = replace(card, volumes=volumes)
    if "laterality" not in unavailable:
        alternatives = ("left", "right", "bilateral", "midline", "none")
        replacement = next(value for value in alternatives if value != card.laterality)
        interventions["laterality"] = replace(card, laterality=replacement)
    if "component_count" not in unavailable:
        interventions["component_count"] = replace(card, component_count=card.component_count + 1)
    interventions["referral"] = replace(
        card,
        referral=not card.referral,
        referral_reasons=() if card.referral else ("controlled_probe",),
    )
    return interventions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic evidence-to-findings baseline R0."
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bertscore-model")
    parser.add_argument("--bertscore-device")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()

    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if (
        any(record.get("report_split") == "test" for record in records)
        and not args.unlock_final_test
    ):
        raise SystemExit("Final report test is locked")

    predictions = []
    intervention_results: dict[str, list[float]] = {}
    for record in records:
        card = EvidenceCard.from_dict(record["evidence"])
        text = render_findings(card)
        validate_rendered_values(card, text)
        consistency = evidence_consistency(text, card.to_dict())
        result = {
            "subject_id": record["subject_id"],
            "prediction": text,
            **consistency,
        }
        reference = record.get("report_text")
        if reference is not None:
            result["reference"] = reference
            result["reference_normalized_findings"] = record.get(
                "normalized_findings"
            ) or extract_normalized_findings(reference)
            result["bleu_1"] = bleu(text, reference, 1)
            result["bleu_4"] = bleu(text, reference, 4)
            result["rouge_l"] = rouge_l(text, reference)
        predictions.append(result)

        before = extract_evidence_fields(text)
        for field, changed_card in intervention_cards(card).items():
            after = extract_evidence_fields(render_findings(changed_card))
            score = intervention_accuracy([before], [after], field)
            intervention_results.setdefault(field, []).append(score)

    texts = [record["prediction"] for record in predictions]
    reference_predictions = [record for record in predictions if "reference" in record]
    bertscore_result = None
    if args.bertscore_model and reference_predictions:
        bertscore_result = bertscore(
            [record["prediction"] for record in reference_predictions],
            [record["reference"] for record in reference_predictions],
            model_type=args.bertscore_model,
            device=args.bertscore_device,
        )
        for index, record in enumerate(reference_predictions):
            record["bertscore_precision"] = bertscore_result["precision"][index]
            record["bertscore_recall"] = bertscore_result["recall"][index]
            record["bertscore_f1"] = bertscore_result["f1"][index]
    all_interventions = [
        score for field_scores in intervention_results.values() for score in field_scores
    ]
    summary = {
        "baseline": "R0_deterministic_evidence",
        "subjects": len(predictions),
        "bleu_policy": "clipped_add_one_smoothed_effective_order_v1",
        "bertscore_status": "computed" if bertscore_result else "not_requested",
        "bertscore_model": bertscore_result["model_type"] if bertscore_result else None,
        "bertscore_package_version": (
            bertscore_result["package_version"] if bertscore_result else None
        ),
        "exact_trace_rate": 1.0 if predictions else float("nan"),
        "structured_field_recall": float(
            np.mean([row["structured_field_recall"] for row in predictions])
        ),
        "structured_field_unsupported_rate": float(
            np.mean([row["structured_field_unsupported_rate"] for row in predictions])
        ),
        "structured_field_contradiction_rate": float(
            np.mean([row["structured_field_contradiction_rate"] for row in predictions])
        ),
        "broad_unsupported_claim_rate": None,
        "broad_unsupported_claim_status": (
            "requires_versioned_clinical_parser_or_radiologist_review"
        ),
        "intervention_accuracy": float(np.mean(all_interventions)),
        "intervention_accuracy_by_field": {
            field: float(np.mean(scores)) for field, scores in sorted(intervention_results.items())
        },
        **report_diversity(texts),
    }
    if reference_predictions:
        summary.update(
            structured_finding_metrics(
                [record["prediction"] for record in reference_predictions],
                [record["reference_normalized_findings"] for record in reference_predictions],
            )
        )
        summary["bleu_1"] = float(np.mean([row["bleu_1"] for row in reference_predictions]))
        summary["bleu_4"] = float(np.mean([row["bleu_4"] for row in reference_predictions]))
        summary["rouge_l"] = float(np.mean([row["rouge_l"] for row in reference_predictions]))
        summary["bertscore_precision"] = (
            float(np.mean(bertscore_result["precision"])) if bertscore_result else None
        )
        summary["bertscore_recall"] = (
            float(np.mean(bertscore_result["recall"])) if bertscore_result else None
        )
        summary["bertscore_f1"] = (
            float(np.mean(bertscore_result["f1"])) if bertscore_result else None
        )

    output = Path(args.output)
    atomic_json_dump(predictions, output)
    atomic_json_dump(summary, output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
