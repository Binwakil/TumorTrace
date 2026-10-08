#!/usr/bin/env python
"""Render a subject-free manuscript summary from frozen development artifacts.

This script never reads the master inventory, clinical images, reports, or the locked final-test
partition. Case-level development values are reduced to anonymous cohort aggregates before
plotting. Public labels are descriptive; legacy run IDs remain confined to artifact paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "navy": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "purple": "#6D597A",
    "gray": "#7A7F87",
}


def load_json(relative_path: str) -> Any:
    path = ROOT / relative_path
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        provenance = data.get("evaluation_provenance", {})
        if data.get("locked_test_opened") is True or provenance.get("locked_test_opened") is True:
            raise RuntimeError(f"Refusing to plot an artifact marked as locked-test output: {path}")
    return data


def style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
    axis.tick_params(labelsize=8)


def add_values(axis: plt.Axes, bars: Any, *, digits: int = 3, suffix: str = "") -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{digits}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#18212F",
        )


def collect_results() -> dict[str, Any]:
    nnunet = load_json("artifacts/nnunet_final_val_summary.json")
    d3 = load_json("outputs/D3_zscore/val_aggregate.json")
    fixed = load_json("outputs/M0/val_aggregate.json")
    pcgrad = load_json("outputs/M1/val_aggregate.json")
    uncertainty = load_json("outputs/M1_uncertainty_weighting/val_aggregate.json")
    classifier = load_json("outputs/C0/val_aggregate.json")
    separate = load_json("outputs/separate_encoder_matched_control_ampoff/val_aggregate.json")
    burden = load_json("outputs/burden_mlp_matched/val_aggregate.json")
    met_holdout = load_json("outputs/met_holdout_matched_classifier/test_aggregate.json")
    lodo = {
        cohort: load_json(f"outputs/lodo_{cohort.lower()}_development/test_aggregate.json")
        for cohort in ("GLI", "MEN", "MET")
    }
    uncertainty_selection = load_json("artifacts/development_uncertainty_selection.json")
    calibration = load_json("artifacts/development_final_evidence_calibration.json")
    d3_cases = load_json("outputs/D3_zscore/val_cases.json")

    d3_wt = {
        cohort: float(d3["stratified"]["cohort"][cohort]["dice_WT"])
        for cohort in ("GLI", "MEN", "MET")
    }
    return {
        "segmentation_macro_dice": {
            "nnU-Net\nResEnc-M": float(nnunet["macro_dice"]),
            "Single-task\nSegResNet": float(d3["macro_dice"]),
            "Fixed joint": float(fixed["macro_dice"]),
            "Gradient-\nbalanced": float(pcgrad["macro_dice"]),
            "Uncertainty-\nweighted": float(uncertainty["macro_dice"]),
        },
        "cohort_macro_dice": {
            label: {
                cohort: float(artifact["stratified"]["cohort"][cohort]["macro_dice"])
                for cohort in ("GLI", "MEN", "MET")
            }
            for label, artifact in {
                "Single-task": d3,
                "Gradient-balanced": pcgrad,
                "Separate encoders\n(AMP off)": separate,
                "Burden head": burden,
            }.items()
        },
        "classification": {
            label: {
                "Balanced accuracy": float(artifact["balanced_accuracy"]),
                "Macro F1": float(artifact["macro_f1"]),
                "Macro AUROC": float(artifact["macro_auroc"]),
                "ECE": float(artifact["ece"]),
            }
            for label, artifact in {
                "Classification only": classifier,
                "Gradient-balanced": pcgrad,
                "MET source holdout": met_holdout,
            }.items()
        },
        "lodo_retention_percent": {
            cohort: 100.0 * float(lodo[cohort]["dice_WT"]) / d3_wt[cohort]
            for cohort in ("GLI", "MEN", "MET")
        },
        "failure_detection_auroc": {
            {
                "entropy": "Entropy",
                "mc_dropout10": "MC dropout",
                "deep_ensemble3": "Ensemble",
            }[row["method"]]: float(row["failure_auroc"])
            for row in uncertainty_selection["ranked_candidates"]
        },
        "wt_interval_coverage_90": {
            "Pooled": float(
                calibration["volume_intervals"]["WT"]["0.9"][
                    "empirical_validation_coverage"
                ]
            ),
            **{
                cohort: float(
                    np.mean(
                        [
                            max(
                                0.0,
                                float(row["predicted_volume_ml_WT"])
                                - float(
                                    calibration["volume_intervals"]["WT"]["0.9"][
                                        "residual_quantile_ml"
                                    ]
                                ),
                            )
                            <= float(row["target_volume_ml_WT"])
                            <= float(row["predicted_volume_ml_WT"])
                            + float(
                                calibration["volume_intervals"]["WT"]["0.9"][
                                    "residual_quantile_ml"
                                ]
                            )
                            for row in d3_cases
                            if row["cohort"] == cohort
                        ]
                    )
                )
                for cohort in ("GLI", "MEN", "MET")
            },
        },
        "calibration_temperature": float(calibration["temperature"]),
        "scope": "development-only; no locked final-test data",
    }


def render(results: dict[str, Any], output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), constrained_layout=True)
    figure.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)

    # A — same-panel segmentation comparison.
    axis = axes[0, 0]
    labels = list(results["segmentation_macro_dice"])
    values = list(results["segmentation_macro_dice"].values())
    bars = axis.bar(
        np.arange(len(labels)),
        values,
        color=[COLORS["navy"], COLORS["teal"], COLORS["gold"], COLORS["orange"], COLORS["red"]],
        width=0.72,
    )
    axis.set_ylim(0.70, 0.88)
    axis.set_ylabel("Macro region Dice")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_title("A  Segmentation performance")
    add_values(axis, bars)
    style_axis(axis)

    # B — cohort-level negative transfer using only corrected/matched conditions.
    axis = axes[0, 1]
    conditions = list(results["cohort_macro_dice"])
    cohorts = ("GLI", "MEN", "MET")
    x = np.arange(len(cohorts))
    width = 0.19
    for index, (condition, color) in enumerate(
        zip(conditions, (COLORS["teal"], COLORS["orange"], COLORS["purple"], COLORS["gold"]), strict=True)
    ):
        values = [results["cohort_macro_dice"][condition][cohort] for cohort in cohorts]
        axis.bar(x + (index - 1.5) * width, values, width, label=condition, color=color)
    axis.set_ylim(0.50, 0.92)
    axis.set_xticks(x, cohorts)
    axis.set_ylabel("Macro region Dice")
    axis.set_title("B  Cohort-level negative transfer")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    style_axis(axis)

    # C — classification quality and calibration.
    axis = axes[0, 2]
    conditions = list(results["classification"])
    metrics = ("Balanced accuracy", "Macro F1", "Macro AUROC", "ECE")
    x = np.arange(len(metrics))
    width = 0.24
    for index, (condition, color) in enumerate(
        zip(conditions, (COLORS["navy"], COLORS["orange"], COLORS["red"]), strict=True)
    ):
        values = [results["classification"][condition][metric] for metric in metrics]
        axis.bar(x + (index - 1) * width, values, width, label=condition, color=color)
    axis.axhline(0.05, color=COLORS["gray"], linestyle=":", linewidth=1.2)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x, ["Balanced\naccuracy", "Macro F1", "Macro\nAUROC", "ECE"])
    axis.set_title("C  Classification and calibration")
    axis.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=3,
    )
    style_axis(axis)

    # D — LODO whole-tumor retention.
    axis = axes[1, 0]
    labels = list(results["lodo_retention_percent"])
    values = list(results["lodo_retention_percent"].values())
    bars = axis.bar(labels, values, color=[COLORS["teal"], COLORS["gold"], COLORS["red"]])
    axis.axhspan(80, 105, color=COLORS["teal"], alpha=0.08)
    axis.axhline(80, color=COLORS["navy"], linestyle="--", linewidth=1.2, label="80% gate")
    axis.set_ylim(60, 102)
    axis.set_ylabel("IID WT Dice retained (%)")
    axis.set_title("D  Leave-one-disease-out transfer")
    add_values(axis, bars, digits=1, suffix="%")
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    # E — uncertainty/error association.
    axis = axes[1, 1]
    labels = list(results["failure_detection_auroc"])
    values = list(results["failure_detection_auroc"].values())
    bars = axis.bar(labels, values, color=[COLORS["navy"], COLORS["teal"], COLORS["gold"], COLORS["red"]])
    axis.axhline(0.5, color=COLORS["gray"], linestyle=":", linewidth=1.2, label="Chance")
    axis.axhline(0.8, color=COLORS["navy"], linestyle="--", linewidth=1.2, label="Required gate")
    axis.set_ylim(0, 0.9)
    axis.set_ylabel("Failure-detection AUROC")
    axis.set_title("E  Matched failure detection")
    add_values(axis, bars)
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    # F — pooled calibration can conceal cohort failure.
    axis = axes[1, 2]
    labels = list(results["wt_interval_coverage_90"])
    values = list(results["wt_interval_coverage_90"].values())
    bars = axis.bar(labels, values, color=[COLORS["navy"], COLORS["teal"], COLORS["gold"], COLORS["red"]])
    axis.axhspan(0.85, 0.95, color=COLORS["teal"], alpha=0.10, label="Allowed ±5 points")
    axis.axhline(0.90, color=COLORS["navy"], linestyle="--", linewidth=1.2, label="Nominal 90%")
    axis.set_ylim(0.78, 1.03)
    axis.set_ylabel("Empirical WT interval coverage")
    axis.set_title("F  Cohortwise volume-interval coverage")
    add_values(axis, bars)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(axis)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-stem",
        default="Manuscript/Figures/figure2_development_results",
        help="Output path without extension; PDF, PNG, and JSON are written.",
    )
    args = parser.parse_args()
    output_stem = ROOT / args.output_stem
    results = collect_results()
    render(results, output_stem)
    output_stem.with_suffix(".json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
