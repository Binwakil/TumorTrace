#!/usr/bin/env python
"""Render the matched development-only 7B decoder ablation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/development_vlm_decoder_comparison.json"
MODE_LABELS = {
    "text_only": "Text",
    "evidence_only": "Evidence",
    "full": "Visual + evidence",
}
MODEL_LABELS = {
    "LLaVA-v1.5/Vicuna-7B": "Vicuna-7B",
    "LLaVA-Med-v1.5/Mistral-7B": "LLaVA-Med",
}
MODE_COLORS = {
    "text_only": "#7A5195",
    "evidence_only": "#009E73",
    "full": "#D55E00",
}
INK = "#172033"
FAIL = "#B42318"


def horizontal_metric(
    axis: plt.Axes,
    rows: list[dict],
    labels: list[str],
    key: str,
    panel_label: str,
    gate: float | None,
    *,
    higher_is_better: bool,
    limit: float,
    show_labels: bool = True,
) -> None:
    values = np.asarray(
        [row[key] if row.get(key) is not None else np.nan for row in rows], dtype=float
    )
    positions = np.arange(len(rows))
    gate_text = None
    if gate is not None and higher_is_better:
        axis.axvspan(0, gate, color="#FFF1F0", zorder=0)
        gate_text = f"required ≥ {gate:.0%}"
    elif gate is not None:
        axis.axvspan(gate, limit, color="#FFF1F0", zorder=0)
        gate_text = f"required ≤ {gate:.0%}"
    colors = [MODE_COLORS[row["mode"]] for row in rows]
    bars = axis.barh(positions, np.nan_to_num(values), color=colors, height=0.57, zorder=2)
    for bar, value in zip(bars, values, strict=True):
        if not np.isfinite(value):
            bar.set_alpha(0.0)
            axis.text(
                0.012 * limit,
                bar.get_y() + bar.get_height() / 2,
                "n/a",
                va="center",
                fontsize=7,
            )
        else:
            axis.text(
                min(value + 0.018 * limit, 0.93 * limit),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=7,
            )
    if gate is not None:
        axis.axvline(gate, color=FAIL, linestyle="--", linewidth=1.2, zorder=3)
    axis.set_yticks(positions, labels if show_labels else [""] * len(labels))
    axis.invert_yaxis()
    axis.set_xlim(0, limit)
    axis.set_title(panel_label, loc="left", fontweight="bold", fontsize=8.5, pad=5)
    if gate_text is not None:
        axis.text(
            0.99,
            1.01,
            gate_text,
            transform=axis.transAxes,
            ha="right",
            color=FAIL,
            fontsize=6.1,
        )
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.axhline(2.5, color="#98A2B3", linewidth=0.8)


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload.get("locked_final_test_opened") is not False:
        raise RuntimeError("Reporter comparison is not explicitly development-only")
    rows = payload["rows"]
    if len(rows) != 6 or {row["subjects"] for row in rows} != {69}:
        raise ValueError("Expected six matched conditions on the frozen 69-case panel")
    labels = [
        f"{MODEL_LABELS[row['model']]} · {MODE_LABELS[row['mode']]}" for row in rows
    ]

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK}
    )
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.55))
    figure.subplots_adjust(
        left=0.145,
        right=0.995,
        top=0.88,
        bottom=0.14,
        wspace=0.13,
    )
    horizontal_metric(
        axes[0],
        rows,
        labels,
        "finding_f1",
        "A  Finding F1",
        0.90,
        higher_is_better=True,
        limit=1.0,
    )
    horizontal_metric(
        axes[1],
        rows,
        labels,
        "laterality_f1",
        "B  Laterality F1",
        None,
        higher_is_better=True,
        limit=1.0,
        show_labels=False,
    )
    horizontal_metric(
        axes[2],
        rows,
        labels,
        "unsupported",
        "C  Unsupported fields",
        0.05,
        higher_is_better=False,
        limit=0.50,
        show_labels=False,
    )
    horizontal_metric(
        axes[3],
        rows,
        labels,
        "evidence_intervention_accuracy",
        "D  Evidence intervention",
        0.90,
        higher_is_better=True,
        limit=1.0,
        show_labels=False,
    )

    output = ROOT / "Manuscript/Figures/figure6_reporter_ablation"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    table_dir = ROOT / "Manuscript/Tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_payload = {
        "scope": "development report-validation only",
        "subjects": 69,
        "locked_test_opened": False,
        "source": str(SOURCE.relative_to(ROOT)),
        "rows": rows,
        "decisions": payload["decisions"],
        "retained_learned_vlm": payload["retained_learned_vlm"],
    }
    (table_dir / "reporter_ablation.json").write_text(
        json.dumps(table_payload, indent=2) + "\n", encoding="utf-8"
    )
    header = (
        "| Model | Condition | Finding F1 | Laterality F1 | Unsupported | Contradiction | "
        "Evidence intervention | Dominant template |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        "| {model} | {condition} | {finding:.4f} | {laterality:.4f} | "
        "{unsupported:.4f} | {contradiction:.4f} | {intervention} | {template:.4f} |\n".format(
            model=row["model"],
            condition=MODE_LABELS[row["mode"]],
            finding=row["finding_f1"],
            laterality=row["laterality_f1"],
            unsupported=row["unsupported"],
            contradiction=row["contradiction"],
            intervention=(
                f"{row['evidence_intervention_accuracy']:.4f}"
                if row["evidence_intervention_accuracy"] is not None
                else "N/A"
            ),
            template=row["dominant_template"],
        )
        for row in rows
    )
    (table_dir / "reporter_ablation.md").write_text(
        "# Matched 7B reporter ablation\n\n"
        "Held-out 69-case development report-validation panel; no locked-test data. "
        "Both decoder families were LoRA-fine-tuned before evaluation.\n\n"
        + header
        + body,
        encoding="utf-8",
    )
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
