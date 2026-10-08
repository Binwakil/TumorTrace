#!/usr/bin/env python
"""Summarize matched M5 robustness evaluations on development validation."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONDITIONS = {
    "Missing T1": "outputs/m5_missing_t1n_d3/val_aggregate.json",
    "Missing T1c": "outputs/m5_missing_t1c_d3/val_aggregate.json",
    "Missing T2": "outputs/m5_missing_t2w_d3/val_aggregate.json",
    "Missing FLAIR": "outputs/m5_missing_t2f_d3/val_aggregate.json",
    "Gaussian noise": "outputs/m5_shift_noise_d3/val_aggregate.json",
    "Intensity scale": "outputs/m5_shift_intensity_d3/val_aggregate.json",
    "Bias field": "outputs/m5_shift_bias_field_d3/val_aggregate.json",
    "Lower resolution": "outputs/m5_shift_resolution_d3/val_aggregate.json",
}

MODALITY_DROPOUT_MISSING = {
    "Missing T1": "outputs/m5_moddrop_missing_t1n/val_aggregate.json",
    "Missing T1c": "outputs/m5_moddrop_missing_t1c/val_aggregate.json",
    "Missing T2": "outputs/m5_moddrop_missing_t2w/val_aggregate.json",
    "Missing FLAIR": "outputs/m5_moddrop_missing_t2f/val_aggregate.json",
}


def load(path: str) -> dict:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    provenance = payload.get("evaluation_provenance", {})
    if provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Robustness artifact lacks closed-test provenance: {path}")
    if provenance.get("split") != "val" or provenance.get("subjects") != 248:
        raise ValueError(f"Expected frozen 248-case development validation: {path}")
    return payload


def load_historical_d3_baseline(path: str) -> dict:
    """Validate the pre-provenance D3 aggregate through its frozen config and manifest."""
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    config = json.loads((ROOT / "outputs/D3_zscore/resolved_config.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / config["data"]["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config["evaluation"].get("include_final_test") is not False:
        raise ValueError("Historical D3 config does not prove include_final_test=false")
    if manifest.get("final_test_locked") is not True or manifest["counts"].get("val") != 248:
        raise ValueError("Historical D3 manifest does not prove a locked 248-case validation panel")
    return payload


def summarize_result(condition: str, path: str, result: dict, baseline: dict) -> dict:
    cohort_dice = {
        cohort: result["stratified"]["cohort"][cohort]["macro_dice"]
        for cohort in ("GLI", "MEN", "MET")
    }
    cohort_drop = {
        cohort: baseline["stratified"]["cohort"][cohort]["macro_dice"] - value
        for cohort, value in cohort_dice.items()
    }
    return {
        "condition": condition,
        "source_artifact": path,
        "macro_dice": result["macro_dice"],
        "macro_dice_delta": result["macro_dice"] - baseline["macro_dice"],
        "iid_retention": result["macro_dice"] / baseline["macro_dice"],
        "dice_WT": result["dice_WT"],
        "dice_TC": result["dice_TC"],
        "dice_ET": result["dice_ET"],
        "dice_SNFH": result["dice_SNFH"],
        "small_lesion_recall": result["small_lesion_recall"],
        "cohort_macro_dice": cohort_dice,
        "worst_cohort_macro_dice": min(cohort_dice.values()),
        "maximum_cohort_drop": max(cohort_drop.values()),
    }


def main() -> None:
    baseline_path = "outputs/D3_zscore/val_aggregate.json"
    baseline = load_historical_d3_baseline(baseline_path)
    rows = []
    for condition, path in BASELINE_CONDITIONS.items():
        result = load(path)
        rows.append(summarize_result(condition, path, result, baseline))

    dropout_iid_path = "outputs/m5_modality_dropout_d3/val_aggregate.json"
    dropout_iid = load(dropout_iid_path)
    dropout_iid_summary = summarize_result(
        "IID validation", dropout_iid_path, dropout_iid, baseline
    )
    baseline_missing_by_condition = {
        row["condition"]: row for row in rows if row["condition"].startswith("Missing ")
    }
    dropout_missing_rows = []
    for condition, path in MODALITY_DROPOUT_MISSING.items():
        result = load(path)
        row = summarize_result(condition, path, result, dropout_iid)
        row["macro_dice_delta_vs_d3_iid"] = result["macro_dice"] - baseline["macro_dice"]
        row["paired_delta_vs_no_dropout"] = (
            result["macro_dice"] - baseline_missing_by_condition[condition]["macro_dice"]
        )
        dropout_missing_rows.append(row)
    payload = {
        "scope": "frozen 248-case development validation",
        "baseline": {
            "method": "Single-task SegResNet, z-score configuration",
            "macro_dice": baseline["macro_dice"],
            "source_artifact": baseline_path,
        },
        "locked_test_opened": False,
        "rows": rows,
        "modality_dropout": {
            "training_probability": 0.15,
            "iid": dropout_iid_summary,
            "missing_sequence_rows": dropout_missing_rows,
        },
        "worst_condition": min(rows, key=lambda row: row["macro_dice"])["condition"],
        "minimum_iid_retention": min(row["iid_retention"] for row in rows),
    }
    output = ROOT / "artifacts/development_m5_robustness_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Matched development robustness summary",
        "",
        "Frozen D3 z-score checkpoint; 248-case development validation only. The locked final test remains closed.",
        "",
        "## Frozen D3 checkpoint under perturbation",
        "",
        "| Condition | Macro Dice | Delta | IID retention | WT | TC | ET | SNFH | Worst cohort | Max cohort drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['macro_dice']:.4f} | {row['macro_dice_delta']:+.4f} | "
            f"{row['iid_retention']:.1%} | {row['dice_WT']:.4f} | {row['dice_TC']:.4f} | "
            f"{row['dice_ET']:.4f} | {row['dice_SNFH']:.4f} | "
            f"{row['worst_cohort_macro_dice']:.4f} | {row['maximum_cohort_drop']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Matched 15% modality-dropout checkpoint",
            "",
            (
                f"The dropout-trained model scored {dropout_iid['macro_dice']:.4f} on IID "
                f"validation ({dropout_iid['macro_dice'] - baseline['macro_dice']:+.4f} "
                "versus frozen D3)."
            ),
            "",
            "| Condition | Macro Dice | Dropout-model retention | Delta vs no-dropout under same missing sequence | WT | TC | ET | SNFH |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dropout_missing_rows:
        lines.append(
            f"| {row['condition']} | {row['macro_dice']:.4f} | {row['iid_retention']:.1%} | "
            f"{row['paired_delta_vs_no_dropout']:+.4f} | {row['dice_WT']:.4f} | "
            f"{row['dice_TC']:.4f} | {row['dice_ET']:.4f} | {row['dice_SNFH']:.4f} |"
        )
    table = ROOT / "Manuscript/Tables/m5_robustness.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    print(table)


if __name__ == "__main__":
    main()
