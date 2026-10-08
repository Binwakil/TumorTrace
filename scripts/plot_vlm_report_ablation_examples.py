"""Render matched qualitative examples for all six learned 7B reporter conditions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_qualitative_report_comparison import (
    COLORS,
    ROOT,
    compact_findings,
    crop_bounds,
    normalize,
    text_card,
)

ALIASES = ("GLI-C", "MEN-B", "MET-A")
MODEL_BLOCKS = (
    (
        "Vicuna-7B",
        {
            "text_only": "outputs/llava_v15_vicuna_7b_text_only/val_predictions_unconstrained.json",
            "evidence_only": "outputs/llava_v15_vicuna_7b_evidence_only/val_predictions_unconstrained.json",
            "full": "outputs/llava_v15_vicuna_7b_full/val_predictions_unconstrained.json",
        },
    ),
    (
        "LLaVA-Med",
        {
            "text_only": "outputs/llava_med_v15_mistral_7b_text_only/val_predictions_unconstrained.json",
            "evidence_only": "outputs/llava_med_v15_mistral_7b_evidence_only/val_predictions_unconstrained.json",
            "full": "outputs/llava_med_v15_mistral_7b_full/val_predictions_unconstrained.json",
        },
    ),
)
MODE_STYLES = {
    "text_only": ("#7A5195", "#F6F0FA"),
    "evidence_only": ("#009E73", "#EAF7F2"),
    "full": ("#D55E00", "#FFF0EC"),
}


def load_predictions(path: str) -> dict[str, dict]:
    return {
        row["subject_id"]: row
        for row in json.loads((ROOT / path).read_text(encoding="utf-8"))
    }


def add_image_panel(axis: plt.Axes, case: dict, input_dir: Path) -> None:
    data = np.load(input_dir / case["data_file"])
    image = data["image"].astype(np.float32)
    target = np.squeeze(data["target"]).astype(np.uint8)
    prediction = data["prediction"].astype(np.uint8)
    slice_index = int((target > 0).sum(axis=(0, 1)).argmax())
    base = np.rot90(normalize(image[3, :, :, slice_index]))
    target_slice = np.rot90(target[:, :, slice_index])
    prediction_slice = np.rot90(prediction[:, :, slice_index])
    bounds = crop_bounds(base)
    base = base[bounds]
    target_wt = target_slice[bounds] > 0
    predicted_wt = prediction_slice[bounds] > 0

    axis.imshow(base, cmap="gray", vmin=0, vmax=1)
    if target_wt.any():
        axis.contour(target_wt, levels=[0.5], colors=[COLORS["GT"]], linewidths=1.25)
    if predicted_wt.any():
        axis.contour(
            predicted_wt,
            levels=[0.5],
            colors=[COLORS["PRED"]],
            linewidths=1.15,
            linestyles="--",
        )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    for x, text, alignment in ((0.03, "FLAIR", "left"), (0.97, case["alias"], "right")):
        axis.text(
            x,
            0.97,
            text,
            transform=axis.transAxes,
            ha=alignment,
            va="top",
            fontsize=6.2,
            fontweight="bold",
            color="white",
            bbox={"facecolor": COLORS["INK"], "alpha": 0.76, "pad": 1.0, "edgecolor": "none"},
        )


def main() -> None:
    input_dir = ROOT / "artifacts/private/qualitative_development_cases"
    case_index = json.loads((input_dir / "index.json").read_text(encoding="utf-8"))
    if case_index.get("locked_test_opened") is not False:
        raise RuntimeError("Qualitative cases are not explicitly development-only")
    cases = {row["alias"]: row for row in case_index["cases"]}
    selected = [cases[alias] for alias in ALIASES]
    reference_records = {
        row["subject_id"]: row
        for row in json.loads(
            (ROOT / "artifacts/private/reporter_records_val.json").read_text(encoding="utf-8")
        )
    }
    predictions = {
        model: {mode: load_predictions(path) for mode, path in paths.items()}
        for model, paths in MODEL_BLOCKS
    }
    if any(case["subject_id"] not in reference_records for case in selected):
        raise ValueError("Prespecified cases are not all in the frozen report-validation panel")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": COLORS["INK"]})
    figure, axes = plt.subplots(
        6,
        5,
        figsize=(10.8, 10.6),
        gridspec_kw={"width_ratios": (0.75, 1.15, 1.15, 1.15, 1.15)},
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.997,
        top=0.945,
        bottom=0.045,
        wspace=0.008,
        hspace=0.012,
    )

    public_rows = []
    for block_index, (model, _paths) in enumerate(MODEL_BLOCKS):
        for case_index_in_block, case in enumerate(selected):
            row = block_index * len(selected) + case_index_in_block
            subject_id = case["subject_id"]
            add_image_panel(axes[row, 0], case, input_dir)

            reference_axis = axes[row, 1]
            reference_axis.axis("off")
            text_card(
                reference_axis,
                compact_findings(reference_records[subject_id]["normalized_findings"], max_fields=4),
                title="Reference · rule-extracted",
                edge=COLORS["REFERENCE"],
                face="#F5F7FA",
                fontsize=5.15,
                width=26,
            )

            public_conditions = {}
            for column, mode in enumerate(("text_only", "evidence_only", "full"), start=2):
                prediction = predictions[model][mode][subject_id]
                unsupported = prediction["structured_field_unsupported_rate"]
                contradiction = prediction["structured_field_contradiction_rate"]
                edge, face = MODE_STYLES[mode]
                axis = axes[row, column]
                axis.axis("off")
                text_card(
                    axis,
                    compact_findings(prediction["predicted_normalized_findings"], max_fields=4),
                    title=f"Unsupported {unsupported:.0%} · contradiction {contradiction:.0%}",
                    edge=edge,
                    face=face,
                    fontsize=4.95,
                    width=25,
                )
                public_conditions[mode] = {
                    "unsupported": unsupported,
                    "contradiction": contradiction,
                    "normalized_findings": prediction["predicted_normalized_findings"],
                }
            public_rows.append(
                {
                    "model": model,
                    "alias": case["alias"],
                    "cohort": case["cohort"],
                    "conditions": public_conditions,
                }
            )

    column_titles = (
        "MRI + WT contours",
        "Reference facts",
        "Text only",
        "Evidence only",
        "Visual + evidence",
    )
    for axis, title in zip(axes[0], column_titles, strict=True):
        axis.set_title(title, fontsize=8.2, fontweight="bold", pad=7)

    for block_index, (model, _paths) in enumerate(MODEL_BLOCKS):
        top = axes[block_index * 3, 0].get_position().y1
        bottom = axes[block_index * 3 + 2, 0].get_position().y0
        figure.text(
            0.012,
            (top + bottom) / 2,
            model,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.0,
            fontweight="bold",
            color="#3569C8" if block_index == 0 else "#008F80",
        )
    separator_y = (
        axes[2, 0].get_position().y0 + axes[3, 0].get_position().y1
    ) / 2
    figure.add_artist(
        Line2D(
            [0.04, 0.997],
            [separator_y, separator_y],
            transform=figure.transFigure,
            color="#98A2B3",
            linewidth=0.8,
        )
    )
    figure.legend(
        handles=(
            Line2D([0], [0], color=COLORS["GT"], lw=2, label="Reference WT"),
            Line2D([0], [0], color=COLORS["PRED"], lw=2, ls="--", label="Predicted WT"),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=1.8,
        columnspacing=1.2,
    )

    output = ROOT / "Manuscript/Figures/figureS3_vlm_report_ablation_comparison"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "scope": "development report-validation only",
                "subjects_in_panel": 3,
                "report_validation_subjects": 69,
                "locked_test_opened": False,
                "selection": (
                    "same cases as Figure 5; frozen by qualitative_policy_v1 before reporter-output inspection"
                ),
                "training": "all six 7B conditions were LoRA-fine-tuned before matched evaluation",
                "rows": public_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.with_suffix(".pdf"))
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
