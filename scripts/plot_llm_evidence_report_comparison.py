"""Render a compact post-hoc qualitative comparison of evidence reporters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_qualitative_report_comparison import (
    ALIASES,
    COLORS,
    compact_findings,
    crop_bounds,
    normalize,
    text_card,
)

from tumortrust_vlm.evaluation.reporting import evidence_consistency, extract_evidence_fields

GPT_COLOR = "#2F6BBD"


def load_predictions(path: str) -> dict[str, dict]:
    return {
        row["subject_id"]: row
        for row in json.loads((ROOT / path).read_text(encoding="utf-8"))
    }


def compact_evidence(text: str) -> str:
    fields = extract_evidence_fields(text)
    lines = []
    if fields.get("family"):
        lines.append(f"Family  {fields['family']} · model-predicted")
    volumes = [
        f"{region} {fields[f'volume_{region}']:.3f} mL"
        for region in ("WT", "TC", "ET", "SNFH")
        if f"volume_{region}" in fields
    ]
    if volumes:
        lines.append("Volumes  " + " · ".join(volumes[:2]))
        if len(volumes) > 2:
            lines.append("         " + " · ".join(volumes[2:]))
    spatial = []
    if fields.get("laterality") is not None:
        spatial.append(str(fields["laterality"]))
    if fields.get("component_count") is not None:
        spatial.append(f"{fields['component_count']} component(s)")
    if spatial:
        lines.append("Distribution  " + " · ".join(spatial))
    return "\n".join(lines) or "No structured evidence fields expressed"


def add_image(axis: plt.Axes, case: dict, input_dir: Path) -> None:
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
        axis.contour(target_wt, levels=[0.5], colors=[COLORS["GT"]], linewidths=1.35)
    if predicted_wt.any():
        axis.contour(
            predicted_wt,
            levels=[0.5],
            colors=[COLORS["PRED"]],
            linewidths=1.25,
            linestyles="--",
        )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.03,
        0.97,
        "FLAIR",
        transform=axis.transAxes,
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="white",
        bbox={"facecolor": COLORS["INK"], "alpha": 0.75, "pad": 1.0, "edgecolor": "none"},
    )
    axis.text(
        0.97,
        0.97,
        case["alias"],
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="white",
        bbox={"facecolor": COLORS["INK"], "alpha": 0.75, "pad": 1.0, "edgecolor": "none"},
    )


def main() -> None:
    input_dir = ROOT / "artifacts/private/qualitative_development_cases"
    case_index = json.loads((input_dir / "index.json").read_text(encoding="utf-8"))
    if case_index.get("locked_test_opened") is not False:
        raise RuntimeError("Qualitative cases are not explicitly development-only")
    cases = {row["alias"]: row for row in case_index["cases"]}
    records = {
        row["subject_id"]: row
        for row in json.loads(
            (ROOT / "artifacts/private/reporter_records_val.json").read_text(encoding="utf-8")
        )
    }
    paths = {
        "deterministic": "outputs/reporter_deterministic_evidence/val_predictions.json",
        "llava_med": (
            "outputs/llava_med_v15_mistral_7b_evidence_only/"
            "val_predictions_unconstrained.json"
        ),
        "gpt": "outputs/gpt_5_6_sol_evidence_only/val_predictions_unconstrained.json",
    }
    predictions = {name: load_predictions(path) for name, path in paths.items()}
    selected = [cases[alias] for alias in ALIASES]
    if any(case["subject_id"] not in records for case in selected):
        raise ValueError("Prespecified cases are not all in report validation")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": COLORS["INK"]})
    figure, axes = plt.subplots(
        3,
        5,
        figsize=(12.0, 4.7),
        gridspec_kw={"width_ratios": [0.92, 1.12, 1.15, 1.15, 1.15]},
    )
    figure.subplots_adjust(
        left=0.012,
        right=0.997,
        top=0.905,
        bottom=0.052,
        wspace=0.012,
        hspace=0.012,
    )
    headings = (
        "MRI + WT contours",
        "Reference facts",
        "Deterministic renderer",
        "LLaVA-Med evidence",
        "GPT-5.6 evidence",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading, fontsize=7.4, fontweight="bold", pad=4)

    public_cases = []
    for row_index, case in enumerate(selected):
        subject_id = case["subject_id"]
        record = records[subject_id]
        add_image(axes[row_index, 0], case, input_dir)
        for axis in axes[row_index, 1:]:
            axis.axis("off")
        text_card(
            axes[row_index, 1],
            compact_findings(record["normalized_findings"]),
            title="Rule-extracted · unreviewed",
            edge=COLORS["REFERENCE"],
            face="#F5F7FA",
            fontsize=5.8,
            width=27,
        )
        public_record = {"alias": case["alias"], "cohort": case["cohort"], "outputs": {}}
        card_styles = {
            "deterministic": (COLORS["SAFE"], "#EAF7F2"),
            "llava_med": (COLORS["CAUTION"], "#FFF7E8"),
            "gpt": (GPT_COLOR, "#EDF4FF"),
        }
        for column, name in enumerate(("deterministic", "llava_med", "gpt"), start=2):
            prediction = predictions[name][subject_id]
            consistency = evidence_consistency(prediction["prediction"], record["evidence"])
            title = (
                f"Recall {consistency['structured_field_recall']:.0%} · "
                f"U {consistency['structured_field_unsupported_rate']:.0%} · "
                f"C {consistency['structured_field_contradiction_rate']:.0%}"
            )
            edge, face = card_styles[name]
            text_card(
                axes[row_index, column],
                compact_evidence(prediction["prediction"]),
                title=title,
                edge=edge,
                face=face,
                fontsize=5.65,
                width=29,
            )
            public_record["outputs"][name] = {
                "normalized_evidence": extract_evidence_fields(prediction["prediction"]),
                **consistency,
            }
        public_cases.append(public_record)

    figure.legend(
        handles=[
            Line2D([0], [0], color=COLORS["GT"], lw=1.5, label="Reference WT"),
            Line2D(
                [0],
                [0],
                color=COLORS["PRED"],
                lw=1.4,
                ls="--",
                label="Predicted WT",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.012, 0.002),
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handlelength=2.2,
    )
    output = ROOT / "Manuscript/Figures/figureS4_llm_evidence_report_comparison"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    payload = {
        "scope": "post-hoc development report-validation only",
        "selection": "same cases frozen for Figures 3 and 5 before reporter-output inspection",
        "locked_final_test_opened_for_comparison": False,
        "columns": list(headings),
        "aggregate_source": "artifacts/development_evidence_reporter_comparison_v2.json",
        "cases": public_cases,
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
