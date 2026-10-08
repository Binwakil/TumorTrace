#!/usr/bin/env python3
"""Render the aggregate-only locked-final manuscript summary figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAVY = "#244a73"
TEAL = "#148f8b"
GOLD = "#d99a26"
RED = "#c84a4a"
GRAY = "#68737d"
LIGHT = "#e8edf2"


def interval(metric: dict[str, object]) -> tuple[float, float, float]:
    lo, hi = metric["ci95"]
    return float(metric["estimate"]), float(lo), float(hi)


def point_ci(ax: plt.Axes, y: float, metric: dict[str, object], color: str, label: str | None) -> None:
    estimate, lo, hi = interval(metric)
    ax.errorbar(
        estimate,
        y,
        xerr=[[estimate - lo], [hi - estimate]],
        fmt="o",
        color=color,
        ecolor=color,
        capsize=2.5,
        markersize=5,
        linewidth=1.5,
        label=label,
        zorder=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="artifacts/locked_final_summary.json")
    parser.add_argument("--output-prefix", default="Manuscript/Figures/figure2_locked_final_results")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    data = json.loads(summary_path.read_text())
    seg = data["segmentation"]
    evidence = data["evidence_calibration"]["volume_interval_coverage"]
    report = data["deterministic_reporting"]["metrics"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.6), constrained_layout=True)

    # A: final segmentation point estimates and patient-bootstrap intervals.
    ax = axes[0, 0]
    metrics = ["macro_dice", "dice_WT", "dice_TC", "dice_ET", "dice_SNFH"]
    labels = ["Macro", "WT", "TC", "ET", "SNFH"]
    y = np.arange(len(metrics))[::-1]
    for idx, (metric, ypos) in enumerate(zip(metrics, y, strict=True)):
        point_ci(ax, ypos + 0.11, seg["nnunet_resenc"][metric], NAVY, "nnU-Net ResEnc-M" if idx == 0 else None)
        point_ci(ax, ypos - 0.11, seg["segresnet_d3"][metric], TEAL, "Single-task SegResNet" if idx == 0 else None)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.60, 0.94)
    ax.set_xlabel("Dice (95% patient-bootstrap CI)")
    ax.set_title("A  Held-out lockbox segmentation")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)

    # B: prespecified paired improvements.
    ax = axes[0, 1]
    diffs = seg["nnunet_minus_segresnet_paired"]
    y = np.arange(len(metrics))[::-1]
    for metric, ypos in zip(metrics, y, strict=True):
        paired = diffs[metric]
        estimate = paired["mean_difference_left_minus_right"]
        lo, hi = paired["ci95"]
        ax.errorbar(
            estimate,
            ypos,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="o",
            color=GOLD,
            ecolor=GOLD,
            capsize=2.5,
            markersize=5,
            linewidth=1.5,
            zorder=3,
        )
    ax.axvline(0.0, color=GRAY, linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.005, 0.075)
    ax.set_xlabel("Paired Dice difference (nnU-Net − SegResNet)")
    ax.set_title("B  Paired model contrast")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.text(
        0.98,
        0.04,
        "Holm p=0.005 for all",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=7.2,
    )

    # C: final cohort performance and the subgroup gap.
    ax = axes[1, 0]
    cohorts = ["GLI", "MEN", "MET"]
    x = np.arange(3)
    macro = [seg["nnunet_by_cohort"][c]["macro_dice"] for c in cohorts]
    wt = [seg["nnunet_by_cohort"][c]["dice_WT"] for c in cohorts]
    ax.scatter(x - 0.08, macro, color=NAVY, s=26, label="Macro Dice", zorder=3)
    ax.scatter(x + 0.08, wt, color=TEAL, s=26, label="WT Dice", zorder=3)
    ax.set_xticks(x, [f"{c}\n(n={seg['nnunet_by_cohort'][c]['subjects']})" for c in cohorts])
    ax.set_ylim(0.70, 0.95)
    ax.set_ylabel("Dice")
    ax.set_title("C  Cohort performance")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(loc="lower left", frameon=False, ncols=2)

    # D: retained evidence outputs and explicitly disabled referral.
    ax = axes[1, 1]
    regions = ["WT", "TC", "ET", "SNFH"]
    x = np.arange(4)
    cov90 = [evidence[r]["0.9"]["empirical_coverage"] for r in regions]
    cov95 = [evidence[r]["0.95"]["empirical_coverage"] for r in regions]
    ax.scatter(x - 0.08, cov90, color=TEAL, s=26, label="90% interval", zorder=3)
    ax.scatter(x + 0.08, cov95, color=NAVY, s=26, label="95% interval", zorder=3)
    ax.axhline(0.90, color=TEAL, linestyle=":", linewidth=1.0)
    ax.axhline(0.95, color=NAVY, linestyle=":", linewidth=1.0)
    ax.set_xticks(x, regions)
    ax.set_ylim(0.82, 1.005)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("D  D3 evidence calibration and rendering")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(loc="lower center", frameon=False, ncols=2, fontsize=6.8)
    recall = report["structured_field_recall"]["estimate"]
    unsupported = report["structured_field_unsupported_rate"]["estimate"]
    ax.text(
        0.01,
        0.98,
        f"Report n=141  ·  field recall {recall:.3f}\nunsupported {unsupported:.3f}  ·  referral disabled",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=RED,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "boxstyle": "round,pad=0.25"},
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    provenance = {
        "scope": "aggregate-only one-shot locked final",
        "source": str(summary_path),
        "protocol_sha256": data["protocol_sha256"],
        "structured_subjects": 320,
        "report_subjects": data["deterministic_reporting"]["subjects"],
        "contains_subject_level_data": False,
        "figure_level_title_or_caption_embedded": False,
    }
    prefix.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
