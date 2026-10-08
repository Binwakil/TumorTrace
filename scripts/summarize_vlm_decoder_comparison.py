#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tumortrust_vlm.reporting.findings import structured_finding_metrics
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

MODE_LABELS = {
    "text_only": "Text only",
    "evidence_only": "Structured evidence only",
    "full": "3D visual tokens + structured evidence",
}
MODELS = {
    "llava_v15_vicuna_7b": "LLaVA-v1.5/Vicuna-7B",
    "llava_med_v15_mistral_7b": "LLaVA-Med-v1.5/Mistral-7B",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_bootstrap(left: list[dict], right: list[dict], samples: int = 2000) -> dict:
    if [row["subject_id"] for row in left] != [row["subject_id"] for row in right]:
        raise ValueError("Paired VLM predictions have different subject order")
    references = [row["reference_normalized_findings"] for row in left]
    left_text = [row["prediction"] for row in left]
    right_text = [row["prediction"] for row in right]
    rng = np.random.default_rng(20260822)
    f1_delta = []
    unsupported_delta = []
    count = len(left)
    for _ in range(samples):
        indices = rng.integers(0, count, size=count)
        reference_sample = [references[index] for index in indices]
        left_sample = [left_text[index] for index in indices]
        right_sample = [right_text[index] for index in indices]
        left_f1 = structured_finding_metrics(left_sample, reference_sample)[
            "structured_finding_micro_f1"
        ]
        right_f1 = structured_finding_metrics(right_sample, reference_sample)[
            "structured_finding_micro_f1"
        ]
        f1_delta.append(float(left_f1) - float(right_f1))
        unsupported_delta.append(
            float(
                np.mean(
                    [left[index]["structured_field_unsupported_rate"] for index in indices]
                )
                - np.mean(
                    [right[index]["structured_field_unsupported_rate"] for index in indices]
                )
            )
        )
    return {
        "f1_delta_left_minus_right": float(np.mean(f1_delta)),
        "f1_delta_95ci": np.quantile(f1_delta, [0.025, 0.975]).tolist(),
        "unsupported_delta_left_minus_right": float(np.mean(unsupported_delta)),
        "unsupported_delta_95ci": np.quantile(unsupported_delta, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
        "seed": 20260822,
    }


def main() -> None:
    rows = []
    predictions: dict[tuple[str, str], list[dict]] = {}
    for slug, public_name in MODELS.items():
        for mode, mode_label in MODE_LABELS.items():
            root = ROOT / "outputs" / f"{slug}_{mode}"
            unconstrained_path = root / "val_predictions_unconstrained.summary.json"
            constrained_path = root / "val_predictions_constrained.summary.json"
            prediction_path = root / "val_predictions_unconstrained.json"
            probe_path = root / "val_input_probes.summary.json"
            summary = load(unconstrained_path)
            constrained = load(constrained_path)
            probe = load(probe_path)
            predictions[(slug, mode)] = load(prediction_path)
            rows.append(
                {
                    "model": public_name,
                    "model_slug": slug,
                    "mode": mode,
                    "mode_label": mode_label,
                    "subjects": summary["subjects"],
                    "finding_f1": summary["structured_finding_micro_f1"],
                    "laterality_f1": summary["structured_laterality_micro_f1"],
                    "unsupported": summary["structured_field_unsupported_rate"],
                    "contradiction": summary["structured_field_contradiction_rate"],
                    "dominant_template": summary["dominant_template_fraction"],
                    "schema_prefixed_unsupported": constrained[
                        "structured_field_unsupported_rate"
                    ],
                    "schema_prefixed_contradiction": constrained[
                        "structured_field_contradiction_rate"
                    ],
                    "evidence_intervention_accuracy": probe[
                        "targeted_evidence_intervention_accuracy"
                    ],
                    "image_swap_change": probe.get("image_swap_output_change_fraction"),
                    "zero_visual_change": probe.get("zero_image_output_change_fraction"),
                    "bleu_1": summary["bleu_1"],
                    "bleu_4": summary["bleu_4"],
                    "rouge_l": summary["rouge_l"],
                    "source_artifacts": {
                        "unconstrained_summary": str(unconstrained_path.relative_to(ROOT)),
                        "constrained_summary": str(constrained_path.relative_to(ROOT)),
                        "probe_summary": str(probe_path.relative_to(ROOT)),
                        "prediction_sha256": sha256_file(prediction_path),
                    },
                }
            )

    decisions = {}
    for slug, public_name in MODELS.items():
        by_mode = {row["mode"]: row for row in rows if row["model_slug"] == slug}
        full = by_mode["full"]
        evidence = by_mode["evidence_only"]
        versus_text = paired_bootstrap(
            predictions[(slug, "full")], predictions[(slug, "text_only")]
        )
        versus_evidence = paired_bootstrap(
            predictions[(slug, "full")], predictions[(slug, "evidence_only")]
        )
        relative_unsupported_reduction = (
            (evidence["unsupported"] - full["unsupported"]) / evidence["unsupported"]
            if evidence["unsupported"]
            else 0.0
        )
        gates = {
            "finding_f1_at_least_0_90": full["finding_f1"] >= 0.90,
            "unsupported_at_most_0_05": full["unsupported"] <= 0.05,
            "contradiction_at_most_0_02": full["contradiction"] <= 0.02,
            "evidence_intervention_at_least_0_90": (
                full["evidence_intervention_accuracy"] is not None
                and full["evidence_intervention_accuracy"] >= 0.90
            ),
            "full_minus_text_f1_at_least_0_10": (
                versus_text["f1_delta_left_minus_right"] >= 0.10
            ),
            "dominant_template_at_most_0_10": full["dominant_template"] <= 0.10,
            "full_vs_evidence_prespecified_benefit": (
                (
                    versus_evidence["f1_delta_left_minus_right"] >= 0.02
                    and versus_evidence["f1_delta_95ci"][0] > 0
                )
                or (
                    relative_unsupported_reduction >= 0.20
                    and versus_evidence["unsupported_delta_95ci"][1] < 0
                )
            ),
        }
        decisions[public_name] = {
            "gates": gates,
            "all_primary_gates_pass": all(gates.values()),
            "full_vs_text": versus_text,
            "full_vs_evidence": versus_evidence,
            "relative_unsupported_reduction_vs_evidence": relative_unsupported_reduction,
        }
    result = {
        "protocol": "configs/vlm_decoder_comparison.yaml",
        "protocol_sha256": sha256_file(ROOT / "configs/vlm_decoder_comparison.yaml"),
        "evaluation_amendment": "configs/vlm_decoder_evaluation_amendment.yaml",
        "evaluation_amendment_sha256": sha256_file(
            ROOT / "configs/vlm_decoder_evaluation_amendment.yaml"
        ),
        "split": "development_report_validation",
        "locked_final_test_opened": False,
        "rows": rows,
        "decisions": decisions,
        "retained_learned_vlm": any(
            value["all_primary_gates_pass"] for value in decisions.values()
        ),
    }
    output = ROOT / "artifacts/development_vlm_decoder_comparison.json"
    atomic_json_dump(result, output)

    table = [
        "# Matched 7B decoder comparison",
        "",
        "Development report-validation only (n=69); the locked final test was not opened.",
        "",
        "| Model | Input condition | Finding F1 | Laterality F1 | Unsupported | Contradiction | Evidence intervention | Dominant template |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        intervention = row["evidence_intervention_accuracy"]
        table.append(
            f"| {row['model']} | {row['mode_label']} | {row['finding_f1']:.4f} | "
            f"{row['laterality_f1']:.4f} | {row['unsupported']:.4f} | "
            f"{row['contradiction']:.4f} | "
            f"{intervention:.4f} | {row['dominant_template']:.4f} |"
            if intervention is not None
            else f"| {row['model']} | {row['mode_label']} | {row['finding_f1']:.4f} | "
            f"{row['laterality_f1']:.4f} | {row['unsupported']:.4f} | "
            f"{row['contradiction']:.4f} | — | {row['dominant_template']:.4f} |"
        )
    table_path = ROOT / "Manuscript/Tables/vlm_decoder_comparison.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
