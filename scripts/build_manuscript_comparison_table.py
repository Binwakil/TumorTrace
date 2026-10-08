#!/usr/bin/env python
"""Build protocol-tiered manuscript comparison tables from frozen aggregates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Manuscript" / "Tables" / "comparison_context"


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("evaluation_provenance", {})
    if payload.get("locked_test_opened") is True or provenance.get("locked_test_opened") is True:
        raise RuntimeError(f"Refusing locked-test artifact: {path}")
    return payload


def load_locked_aggregate(relative: str) -> dict[str, Any]:
    """Load an already-opened aggregate without reading case-level lockbox data."""
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    scope = str(payload.get("scope", ""))
    provenance = payload.get("evaluation_provenance", {})
    if "locked" not in relative and "locked" not in scope:
        raise RuntimeError(f"Expected a locked aggregate: {path}")
    if provenance and provenance.get("locked_test_opened") is not True:
        raise RuntimeError(f"Locked aggregate lacks an opened-test receipt: {path}")
    return payload


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> None:
    locked = load_locked_aggregate("artifacts/locked_final_summary.json")
    locked_nnunet = load_locked_aggregate("artifacts/locked_final/nnunet_summary.json")
    nnunet = load("artifacts/nnunet_final_val_summary.json")
    segresnet = load("outputs/D3_zscore/val_aggregate.json")
    classifier = load("outputs/C0/val_aggregate.json")
    gradient_balanced = load("outputs/M1/val_aggregate.json")
    met_holdout = load("outputs/met_holdout_matched_classifier/test_aggregate.json")
    deterministic_reporter = load(
        "outputs/reporter_deterministic_evidence/val_predictions.summary.json"
    )
    compact_evidence_reporter = load(
        "outputs/reporter_evidence_only_compact/val_predictions_unconstrained.summary.json"
    )
    llava_med_evidence_reporter = load(
        "outputs/llava_med_v15_mistral_7b_evidence_only/val_predictions_unconstrained.summary.json"
    )
    llava_med_full_reporter = load(
        "outputs/llava_med_v15_mistral_7b_full/val_predictions_unconstrained.summary.json"
    )
    llava_med_evidence_clinical = load(
        "artifacts/development_clinical_text_metrics_llava_med_v15_mistral_7b_evidence.json"
    )
    llava_med_full_clinical = load(
        "artifacts/development_clinical_text_metrics_llava_med_v15_mistral_7b_full.json"
    )

    internal = [
        {
            "task": "Segmentation",
            "method": "nnU-Net ResEnc-M",
            "scope": "Frozen 248-case development validation",
            "macro_dice": nnunet["macro_dice"],
            "balanced_accuracy": None,
            "macro_f1": None,
            "macro_auroc": None,
            "ece": None,
            "source_artifact": "artifacts/nnunet_final_val_summary.json",
        },
        {
            "task": "Segmentation",
            "method": "Single-task SegResNet",
            "scope": "Frozen 248-case development validation",
            "macro_dice": segresnet["macro_dice"],
            "balanced_accuracy": None,
            "macro_f1": None,
            "macro_auroc": None,
            "ece": None,
            "source_artifact": "outputs/D3_zscore/val_aggregate.json",
        },
        {
            "task": "Classification",
            "method": "Classification-only 3D CNN",
            "scope": "Frozen 248-case development validation",
            "macro_dice": None,
            "balanced_accuracy": classifier["balanced_accuracy"],
            "macro_f1": classifier["macro_f1"],
            "macro_auroc": classifier["macro_auroc"],
            "ece": classifier["ece"],
            "source_artifact": "outputs/C0/val_aggregate.json",
        },
        {
            "task": "Multitask",
            "method": "Gradient-balanced multitask 3D CNN",
            "scope": "Frozen 248-case development validation",
            "macro_dice": gradient_balanced["macro_dice"],
            "balanced_accuracy": gradient_balanced["balanced_accuracy"],
            "macro_f1": gradient_balanced["macro_f1"],
            "macro_auroc": gradient_balanced["macro_auroc"],
            "ece": gradient_balanced["ece"],
            "source_artifact": "outputs/M1/val_aggregate.json",
        },
        {
            "task": "Classification transfer",
            "method": "Matched MET-source-holdout classifier",
            "scope": "Corrected 248-case development test role",
            "macro_dice": None,
            "balanced_accuracy": met_holdout["balanced_accuracy"],
            "macro_f1": met_holdout["macro_f1"],
            "macro_auroc": met_holdout["macro_auroc"],
            "ece": met_holdout["ece"],
            "source_artifact": "outputs/met_holdout_matched_classifier/test_aggregate.json",
        },
    ]

    reporting_internal = [
        {
            "method": "Deterministic evidence renderer",
            "finding_f1": deterministic_reporter["structured_finding_micro_f1"],
            "bleu_1": deterministic_reporter["bleu_1"],
            "bleu_4": deterministic_reporter["bleu_4"],
            "rouge_l": deterministic_reporter["rouge_l"],
            "unsupported": deterministic_reporter["structured_field_unsupported_rate"],
            "contradiction": deterministic_reporter["structured_field_contradiction_rate"],
            "radgraph": None,
            "ratescore": None,
            "radcliq": None,
            "source_artifact": "outputs/reporter_deterministic_evidence/val_predictions.summary.json",
        },
        {
            "method": "Learned evidence-only compact reporter",
            "finding_f1": compact_evidence_reporter["structured_finding_micro_f1"],
            "bleu_1": compact_evidence_reporter["bleu_1"],
            "bleu_4": compact_evidence_reporter["bleu_4"],
            "rouge_l": compact_evidence_reporter["rouge_l"],
            "unsupported": compact_evidence_reporter["structured_field_unsupported_rate"],
            "contradiction": compact_evidence_reporter["structured_field_contradiction_rate"],
            "radgraph": None,
            "ratescore": None,
            "radcliq": None,
            "source_artifact": "outputs/reporter_evidence_only_compact/val_predictions_unconstrained.summary.json",
        },
        {
            "method": "LLaVA-Med-v1.5/Mistral-7B · structured evidence only",
            "finding_f1": llava_med_evidence_reporter["structured_finding_micro_f1"],
            "bleu_1": llava_med_evidence_reporter["bleu_1"],
            "bleu_4": llava_med_evidence_reporter["bleu_4"],
            "rouge_l": llava_med_evidence_reporter["rouge_l"],
            "unsupported": llava_med_evidence_reporter["structured_field_unsupported_rate"],
            "contradiction": llava_med_evidence_reporter["structured_field_contradiction_rate"],
            "radgraph": llava_med_evidence_clinical["radgraph_simple"],
            "ratescore": llava_med_evidence_clinical["ratescore"],
            "radcliq": llava_med_evidence_clinical["radcliq_v1_raw_mean"],
            "source_artifact": "outputs/llava_med_v15_mistral_7b_evidence_only/val_predictions_unconstrained.summary.json",
        },
        {
            "method": "LLaVA-Med-v1.5/Mistral-7B · visual tokens + evidence",
            "finding_f1": llava_med_full_reporter["structured_finding_micro_f1"],
            "bleu_1": llava_med_full_reporter["bleu_1"],
            "bleu_4": llava_med_full_reporter["bleu_4"],
            "rouge_l": llava_med_full_reporter["rouge_l"],
            "unsupported": llava_med_full_reporter["structured_field_unsupported_rate"],
            "contradiction": llava_med_full_reporter["structured_field_contradiction_rate"],
            "radgraph": llava_med_full_clinical["radgraph_simple"],
            "ratescore": llava_med_full_clinical["ratescore"],
            "radcliq": llava_med_full_clinical["radcliq_v1_raw_mean"],
            "source_artifact": "outputs/llava_med_v15_mistral_7b_full/val_predictions_unconstrained.summary.json",
        },
    ]

    locked_internal = [
        {
            "task": "Segmentation",
            "method": "nnU-Net ResEnc-M",
            "n": 320,
            "values": "Macro Dice 0.8457 (0.8276-0.8617); WT/TC/ET/SNFH 0.8944/0.8963/0.8687/0.7234",
            "qualification": "Retained headline perception model",
        },
        {
            "task": "Segmentation control",
            "method": "Single-task SegResNet",
            "n": 320,
            "values": "Macro Dice 0.8024 (0.7818-0.8222)",
            "qualification": "Standard CNN evidence model, not architectural novelty",
        },
        {
            "task": "Paired segmentation contrast",
            "method": "nnU-Net minus SegResNet",
            "n": 320,
            "values": "+0.0433 macro Dice (0.0324-0.0539); Holm p=0.0050",
            "qualification": "Prespecified paired patient comparison",
        },
        {
            "task": "Tumor-family classification",
            "method": "Classification-only 3D CNN, temperature scaled",
            "n": 320,
            "values": "Bal. acc. 0.8538 (0.8133-0.8902); macro F1 0.8272; AUROC 0.9591; ECE 0.0693",
            "qualification": "Auxiliary/source-limited because shortcut gates failed in development",
        },
        {
            "task": "Structured reporting",
            "method": "Deterministic evidence renderer",
            "n": 141,
            "values": "Field recall 0.9953 (0.9905-0.9988); unsupported 0; contradiction 0",
            "qualification": "Automated schema agreement, not radiologist validation",
        },
    ]

    locked_reporting = locked["deterministic_reporting"]
    locked_clinical = locked_reporting["clinical_text_metrics"]
    locked_report_metrics = locked_reporting["metrics"]

    context = [
        {
            "tier": "Published-protocol context",
            "domain": "Adult glioma segmentation",
            "tumortrust": {
                region: locked_nnunet["stratified"]["cohort"]["GLI"][f"dice_{region}"]
                for region in ("WT", "TC", "ET")
            },
            "reference": {"WT": 0.9005, "TC": 0.8673, "ET": 0.8509},
            "reference_name": "BraTS 2023 adult-glioma winner",
            "reference_url": "https://arxiv.org/abs/2402.17317",
            "qualification": "Different validation split and challenge evaluator; no head-to-head claim.",
        },
        {
            "tier": "Published-protocol context",
            "domain": "Adult glioma segmentation",
            "tumortrust": {"WT": locked_nnunet["stratified"]["cohort"]["GLI"]["dice_WT"]},
            "reference": {"WT": 0.928},
            "reference_name": "DARE-FUSE on BraTS 2021",
            "reference_url": "https://doi.org/10.1038/s41746-026-02365-3",
            "qualification": "Different training data and evaluator; the published WT value is contextual and not a local reproduction.",
        },
        {
            "tier": "Published-protocol context",
            "domain": "Brain metastasis segmentation",
            "tumortrust": {
                region: locked_nnunet["stratified"]["cohort"]["MET"][f"dice_{region}"]
                for region in ("WT", "TC", "ET")
            },
            "reference": {"WT": 0.62, "TC": 0.65, "ET": 0.60},
            "reference_name": "BraTS-METS 2023 winner",
            "reference_url": "https://www.melba-journal.org/2024:001",
            "qualification": "Reference is lesion-wise with explicit FP/FN penalties; local ordinary Dice is not directly comparable.",
        },
        {
            "tier": "Reporting stretch reference",
            "domain": "Grounded brain-MRI reporting",
            "tumortrust": {
                "RadGraph": 100 * locked_clinical["radgraph_simple"]["estimate"],
                "RadCliQ": locked_clinical["radcliq_v1"]["estimate"],
                "RaTEScore": 100 * locked_clinical["ratescore"]["estimate"],
            },
            "reference": {"RadGraph": 28.75, "RadCliQ": 0.54, "RaTEScore": 46.65},
            "reference_name": "AutoRG-Brain, RadGenome-Brain Prompt",
            "reference_url": "https://doi.org/10.1109/TMI.2024.3440351",
            "qualification": "Different split and model-specific metric implementations; RadGraph/RaTEScore are shown as percentages. These chest-derived metrics are secondary and do not establish a head-to-head win.",
        },
        {
            "tier": "Reporting stretch reference",
            "domain": "Structured brain-MRI reporting",
            "tumortrust": {
                "BLEU-1": locked_report_metrics["bleu_1"]["estimate"],
                "ROUGE-L": locked_report_metrics["rouge_l"]["estimate"],
                "RaTEScore": locked_clinical["ratescore"]["estimate"],
            },
            "reference": {
                "BLEU-1": 0.248,
                "ROUGE-1": 0.371,
                "TBFact-F1": 0.359,
                "RaTEScore": 0.577,
            },
            "reference_name": "BTReport on HuskyBrain",
            "reference_url": "https://proceedings.mlr.press/v315/heras-rivera26a.html",
            "qualification": "Dataset-specific stretch reference; not a same-split comparator.",
        },
    ]

    result = {
        "scope": "held-out internal lockbox aggregates plus clearly separated development analyses and published-protocol context",
        "benchmarking_policy": "Same-split rows are authoritative; published rows are protocol-qualified context or stretch references.",
        "held_out_internal_lockbox": locked_internal,
        "internal_same_split": internal,
        "reporting_same_split": reporting_internal,
        "published_context": context,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# TumorTrust-VLM comparison table",
        "",
        "**Scope:** held-out internal lockbox aggregates are separated from development ablations and from published cross-dataset context. Published rows are not head-to-head comparisons.",
        "",
        "## Held-out internal lockbox results",
        "",
        "| Task | Descriptive method | n | Primary values (95% CI) | Qualification |",
        "|---|---|---:|---|---|",
    ]
    for row in locked_internal:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['n']} | {row['values']} | "
            f"{row['qualification']} |"
        )
    lines.extend(
        [
        "",
        "## Same-split development comparisons",
        "",
        "| Task | Descriptive method | Macro Dice | Balanced accuracy | Macro F1 | Macro AUROC | ECE | Traceable artifact |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in internal:
        lines.append(
            f"| {row['task']} | {row['method']} | {fmt(row['macro_dice'])} | "
            f"{fmt(row['balanced_accuracy'])} | {fmt(row['macro_f1'])} | "
            f"{fmt(row['macro_auroc'])} | {fmt(row['ece'])} | `{row['source_artifact']}` |"
        )
    lines.extend(
        [
            "",
            "## Same-split development reporting",
            "",
            "| Descriptive method | Finding F1 | BLEU-1 | BLEU-4 | ROUGE-L | RadGraph | RaTEScore | RadCliQ↓ | Unsupported | Contradiction | Traceable artifact |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in reporting_internal:
        lines.append(
            f"| {row['method']} | {fmt(row['finding_f1'])} | {fmt(row['bleu_1'])} | "
            f"{fmt(row['bleu_4'])} | {fmt(row['rouge_l'])} | {fmt(row['radgraph'])} | "
            f"{fmt(row['ratescore'])} | {fmt(row['radcliq'])} | {fmt(row['unsupported'])} | "
            f"{fmt(row['contradiction'])} | `{row['source_artifact']}` |"
        )
    lines.extend(
        [
            "",
            "## Published-protocol context",
            "",
            "These rows are not head-to-head rankings unless the cited evaluator and split are reproduced.",
            "",
            "| Domain | TumorTrust value | Published reference | Qualification |",
            "|---|---|---|---|",
        ]
    )
    for row in context:
        ours_values = (
            "Pending"
            if row["tumortrust"] is None
            else " / ".join(f"{key} {value:.4f}" for key, value in row["tumortrust"].items())
        )
        cohort = (
            "n=148"
            if row["domain"] == "Adult glioma segmentation"
            else "n=49"
            if row["domain"] == "Brain metastasis segmentation"
            else "n=141"
        )
        ours = f"**Held-out internal lockbox {cohort}:** {ours_values}"
        reference = " / ".join(f"{key} {value}" for key, value in row["reference"].items())
        lines.append(
            f"| {row['domain']} | {ours} | [{row['reference_name']}]({row['reference_url']}): "
            f"{reference} | {row['qualification']} |"
        )
    OUTPUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT.with_suffix(".json"))
    print(OUTPUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
