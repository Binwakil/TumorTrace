#!/usr/bin/env python
"""Render the development-only LODO, missing-sequence, and shift stress tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = "artifacts/development_m5_robustness_summary.json"
LODO_PATHS = {
    "GLI": "outputs/lodo_gli_development/test_aggregate.json",
    "MEN": "outputs/lodo_men_development/test_aggregate.json",
    "MET": "outputs/lodo_met_development/test_aggregate.json",
}
COLORS = {
    "navy": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "red": "#D55E00",
    "gray": "#7A7F87",
}


def load(relative_path: str) -> dict:
    payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    provenance = payload.get("evaluation_provenance", {})
    if payload.get("locked_test_opened") is True or provenance.get("locked_test_opened") is True:
        raise ValueError(f"Refusing locked-test artifact: {relative_path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8)


def label_bars(axis: plt.Axes, bars: object, *, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def main() -> None:
    summary = load(SUMMARY_PATH)
    baseline_rows = {row["condition"]: row for row in summary["rows"]}
    dropout_rows = {
        row["condition"]: row for row in summary["modality_dropout"]["missing_sequence_rows"]
    }
    baseline_iid = float(summary["baseline"]["macro_dice"])
    dropout_iid = float(summary["modality_dropout"]["iid"]["macro_dice"])

    d3 = load("outputs/D3_zscore/val_aggregate.json")
    lodo = {cohort: load(path) for cohort, path in LODO_PATHS.items()}
    retention = {
        cohort: lodo[cohort]["dice_WT"]
        / d3["stratified"]["cohort"][cohort]["dice_WT"]
        for cohort in ("GLI", "MEN", "MET")
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
    figure.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.18, wspace=0.20)

    # A: cohort holdout transfer.
    axis = axes[0]
    cohorts = list(retention)
    values = [retention[cohort] for cohort in cohorts]
    bars = axis.bar(
        np.arange(3),
        values,
        color=[COLORS["teal"] if value >= 0.80 else COLORS["red"] for value in values],
        width=0.68,
    )
    axis.axhline(0.80, color=COLORS["red"], linestyle="--", linewidth=1.1)
    axis.text(2.42, 0.815, "80% gate", ha="right", color=COLORS["red"], fontsize=7)
    axis.set_xticks(np.arange(3), ["GLI\n(n=125)", "MEN\n(n=100)", "MET\n(n=23)"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("WT Dice retention")
    axis.set_title("A  Leave-one-cohort-out")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            fontsize=8,
        )
    style(axis)

    # B: paired missing-sequence response.
    axis = axes[1]
    conditions = ["Missing T1", "Missing T1c", "Missing T2", "Missing FLAIR"]
    x = np.arange(len(conditions))
    width = 0.36
    no_dropout = [baseline_rows[condition]["macro_dice"] for condition in conditions]
    with_dropout = [dropout_rows[condition]["macro_dice"] for condition in conditions]
    left = axis.bar(
        x - width / 2,
        no_dropout,
        width,
        color=COLORS["navy"],
        label=f"No dropout (IID {baseline_iid:.3f})",
    )
    right = axis.bar(
        x + width / 2,
        with_dropout,
        width,
        color=COLORS["gold"],
        label=f"15% dropout (IID {dropout_iid:.3f})",
    )
    axis.set_xticks(x, ["T1", "T1c", "T2", "FLAIR"])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Macro region Dice")
    axis.set_title("B  Single missing sequence")
    axis.legend(frameon=False, fontsize=7, loc="lower left")
    label_bars(axis, left)
    label_bars(axis, right)
    style(axis)

    # C: controlled perturbations on frozen D3.
    axis = axes[2]
    shifts = ["Gaussian noise", "Intensity scale", "Bias field", "Lower resolution"]
    shift_labels = ["Noise", "Intensity", "Bias field", "Lower res."]
    shift_retention = [baseline_rows[condition]["iid_retention"] for condition in shifts]
    positions = np.arange(4)
    axis.scatter(positions, shift_retention, color=COLORS["teal"], s=38, zorder=3)
    axis.axhline(1.0, color=COLORS["gray"], linestyle=":", linewidth=1.1)
    axis.set_xticks(np.arange(4), shift_labels, rotation=20, ha="right")
    axis.set_ylim(0.90, 1.025)
    axis.set_ylabel("Macro Dice / IID Dice")
    axis.set_title("C  Controlled image shifts")
    for position, value in zip(positions, shift_retention, strict=True):
        axis.text(
            position,
            value + 0.003,
            f"{value:.1%}",
            ha="center",
            fontsize=7,
        )
    style(axis)

    output = ROOT / "Manuscript/Figures/figure4_robustness_stress_test"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)

    provenance = {
        "scope": "development-only",
        "locked_test_opened": False,
        "sources": {
            path: sha256_file(ROOT / path)
            for path in [SUMMARY_PATH, "outputs/D3_zscore/val_aggregate.json", *LODO_PATHS.values()]
        },
        "lodo_wt_retention": retention,
        "baseline_iid_macro_dice": baseline_iid,
        "modality_dropout_iid_macro_dice": dropout_iid,
        "missing_sequence_no_dropout_macro_dice": dict(zip(conditions, no_dropout, strict=True)),
        "missing_sequence_dropout_macro_dice": dict(zip(conditions, with_dropout, strict=True)),
        "controlled_shift_retention": dict(zip(shifts, shift_retention, strict=True)),
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
