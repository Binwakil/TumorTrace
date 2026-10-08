"""Build a same-split aggregate comparison of deterministic, 7B, and API reporters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.evaluation.reporting import (
    EVIDENCE_EXTRACTOR_VERSION,
    evidence_consistency,
    report_diversity,
)
from tumortrust_vlm.reporting.findings import (
    structured_finding_metrics,
    structured_laterality_metrics,
)
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

VLM_CONDITIONS = (
    ("LLaVA-v1.5/Vicuna-7B", "Text only", "llava_v15_vicuna_7b_text_only", False),
    ("LLaVA-v1.5/Vicuna-7B", "Structured evidence", "llava_v15_vicuna_7b_evidence_only", False),
    ("LLaVA-v1.5/Vicuna-7B", "Visual + evidence", "llava_v15_vicuna_7b_full", True),
    (
        "LLaVA-Med-v1.5/Mistral-7B",
        "Text only",
        "llava_med_v15_mistral_7b_text_only",
        False,
    ),
    (
        "LLaVA-Med-v1.5/Mistral-7B",
        "Structured evidence",
        "llava_med_v15_mistral_7b_evidence_only",
        False,
    ),
    (
        "LLaVA-Med-v1.5/Mistral-7B",
        "Visual + evidence",
        "llava_med_v15_mistral_7b_full",
        True,
    ),
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def base_metrics(predictions: list[dict], records: dict[str, dict]) -> dict:
    consistency = [
        evidence_consistency(row["prediction"], records[row["subject_id"]]["evidence"])
        for row in predictions
    ]
    texts = [row["prediction"] for row in predictions]
    references = [
        row.get("reference_normalized_findings")
        or records[row["subject_id"]]["normalized_findings"]
        for row in predictions
    ]
    return {
        "structured_field_recall": float(
            np.mean([row["structured_field_recall"] for row in consistency])
        ),
        "unsupported": float(
            np.mean([row["structured_field_unsupported_rate"] for row in consistency])
        ),
        "contradiction": float(
            np.mean([row["structured_field_contradiction_rate"] for row in consistency])
        ),
        **structured_finding_metrics(texts, references),
        **structured_laterality_metrics(texts, references),
        **report_diversity(texts),
    }


def main() -> None:
    records_path = ROOT / "artifacts/private/reporter_records_val.json"
    record_rows = load(records_path)
    if len(record_rows) != 69 or {row.get("report_split") for row in record_rows} != {"val"}:
        raise ValueError("Expected the frozen 69-case development report-validation panel")
    records = {row["subject_id"]: row for row in record_rows}
    rows = []

    deterministic_path = ROOT / "outputs/reporter_deterministic_evidence/val_predictions.json"
    deterministic_summary_path = deterministic_path.with_suffix(".summary.json")
    deterministic = load(deterministic_path)
    deterministic_summary = load(deterministic_summary_path)
    metrics = base_metrics(deterministic, records)
    rows.append(
        {
            "model": "Deterministic evidence renderer",
            "condition": "Structured evidence",
            "training": "None",
            "visual_input": False,
            "subjects": 69,
            "finding_f1": metrics["structured_finding_micro_f1"],
            "laterality_f1": metrics["structured_laterality_micro_f1"],
            "structured_field_recall": metrics["structured_field_recall"],
            "unsupported": metrics["unsupported"],
            "contradiction": metrics["contradiction"],
            "evidence_intervention_accuracy": deterministic_summary["intervention_accuracy"],
            "dominant_template": metrics["dominant_template_fraction"],
            "base_api_cost_usd": 0.0,
            "base_latency_seconds": None,
            "intervention_evaluator": "exact deterministic counterfactual rendering",
            "source_predictions": str(deterministic_path.relative_to(ROOT)),
        }
    )

    for model, condition, slug, visual in VLM_CONDITIONS:
        prediction_path = ROOT / "outputs" / slug / "val_predictions_unconstrained.json"
        summary_path = prediction_path.with_suffix(".summary.json")
        probe_summary_path = ROOT / "outputs" / slug / "val_input_probes.summary.json"
        predictions = load(prediction_path)
        original_summary = load(summary_path)
        probe = load(probe_summary_path)
        metrics = base_metrics(predictions, records)
        rows.append(
            {
                "model": model,
                "condition": condition,
                "training": "LoRA fine-tuned on 482 OOF development reports",
                "visual_input": visual,
                "subjects": 69,
                "finding_f1": metrics["structured_finding_micro_f1"],
                "laterality_f1": metrics["structured_laterality_micro_f1"],
                "structured_field_recall": metrics["structured_field_recall"],
                "unsupported": metrics["unsupported"],
                "contradiction": metrics["contradiction"],
                "evidence_intervention_accuracy": probe.get(
                    "targeted_evidence_intervention_accuracy"
                ),
                "dominant_template": metrics["dominant_template_fraction"],
                "base_api_cost_usd": None,
                "base_latency_seconds": None,
                "intervention_evaluator": (
                    "legacy v1 parser; generated intervention texts were not archived for v2 "
                    "re-scoring"
                ),
                "v1_to_v2_base_consistency_changed": not (
                    np.isclose(
                        original_summary["structured_field_recall"],
                        metrics["structured_field_recall"],
                    )
                    and np.isclose(
                        original_summary["structured_field_unsupported_rate"],
                        metrics["unsupported"],
                    )
                    and np.isclose(
                        original_summary["structured_field_contradiction_rate"],
                        metrics["contradiction"],
                    )
                ),
                "source_predictions": str(prediction_path.relative_to(ROOT)),
            }
        )

    gpt_path = ROOT / "outputs/gpt_5_6_sol_evidence_only/val_predictions_unconstrained.json"
    gpt_summary_path = gpt_path.with_suffix(".summary.json")
    gpt_probe_path = ROOT / "outputs/gpt_5_6_sol_evidence_only/val_input_probes.summary.json"
    gpt_predictions = load(gpt_path)
    gpt_summary = load(gpt_summary_path)
    gpt_probe = load(gpt_probe_path)
    metrics = base_metrics(gpt_predictions, records)
    rows.append(
        {
            "model": "GPT-5.6 Sol",
            "condition": "Structured evidence",
            "training": "Zero-shot API; no study-specific fine-tuning",
            "visual_input": False,
            "subjects": 69,
            "finding_f1": metrics["structured_finding_micro_f1"],
            "laterality_f1": metrics["structured_laterality_micro_f1"],
            "structured_field_recall": metrics["structured_field_recall"],
            "unsupported": metrics["unsupported"],
            "contradiction": metrics["contradiction"],
            "evidence_intervention_accuracy": gpt_probe[
                "targeted_evidence_intervention_accuracy"
            ],
            "dominant_template": metrics["dominant_template_fraction"],
            "base_api_cost_usd": gpt_summary["estimated_api_cost_usd"],
            "base_latency_seconds": gpt_summary["sequential_latency_seconds"],
            "full_audit_api_cost_usd": (
                gpt_summary["estimated_api_cost_usd"] + gpt_probe["estimated_api_cost_usd"]
            ),
            "intervention_evaluator": EVIDENCE_EXTRACTOR_VERSION,
            "source_predictions": str(gpt_path.relative_to(ROOT)),
        }
    )

    result = {
        "scope": "post-hoc development report-validation comparison",
        "subjects": 69,
        "locked_final_test_opened_for_comparison": False,
        "evidence_extractor_version": EVIDENCE_EXTRACTOR_VERSION,
        "protocol": "configs/llm_evidence_reporter.yaml",
        "protocol_sha256": sha256_file(ROOT / "configs/llm_evidence_reporter.yaml"),
        "evaluation_amendment": "configs/llm_evidence_evaluation_amendment.yaml",
        "evaluation_amendment_sha256": sha256_file(
            ROOT / "configs/llm_evidence_evaluation_amendment.yaml"
        ),
        "rows": rows,
        "interpretation": {
            "retained_reporting_path": "Deterministic evidence renderer",
            "natural_language_option": (
                "GPT-5.6 Sol passes automated structured grounding gates as a zero-shot "
                "evidence verbalizer but remains post-hoc, proprietary, non-visual, and not "
                "clinically validated."
            ),
            "learned_7b_decision": (
                "Rejected: high structured contradiction/unsupported rates and negligible "
                "legacy intervention responsiveness."
            ),
        },
    }
    artifact_path = ROOT / "artifacts/development_evidence_reporter_comparison_v2.json"
    atomic_json_dump(result, artifact_path)

    table_path = ROOT / "Manuscript/Tables/evidence_reporter_comparison_v2.md"
    header = (
        "| Model | Input | Training | Finding F1 | Laterality F1 | Evidence recall | "
        "Unsupported | Contradiction | Intervention | Dominant template |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {row['model']} | {row['condition']} | {row['training']} | "
        f"{row['finding_f1']:.4f} | {row['laterality_f1']:.4f} | "
        f"{row['structured_field_recall']:.4f} | {row['unsupported']:.4f} | "
        f"{row['contradiction']:.4f} | "
        + (
            f"{row['evidence_intervention_accuracy']:.4f}"
            if row["evidence_intervention_accuracy"] is not None
            else "—"
        )
        + f" | {row['dominant_template']:.4f} |\n"
        for row in rows
    )
    table_path.write_text(
        "# Same-split evidence reporter comparison\n\n"
        "Post-hoc comparison on the held-out 69-case development report panel; no additional "
        "locked-final access. GPT-5.6 Sol is zero-shot and evidence-only, not a visually grounded "
        "or fine-tuned VLM. Broad free-text safety still requires radiologist review.\n\n"
        + header
        + body,
        encoding="utf-8",
    )
    print(artifact_path)
    print(table_path)


if __name__ == "__main__":
    main()
