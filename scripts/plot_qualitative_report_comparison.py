#!/usr/bin/env python
"""Render a readable, development-only diagnostic report comparison panel."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ("GLI-C", "MEN-B", "MET-A")
COLORS = {
    "INK": "#172033",
    "MUTED": "#667085",
    "GT": "#00B8D9",
    "PRED": "#FFB000",
    "REFERENCE": "#697586",
    "SAFE": "#087E8B",
    "CAUTION": "#3763A0",
    "FAIL": "#7652A3",
}


def normalize(image: np.ndarray) -> np.ndarray:
    foreground = image[np.isfinite(image) & (image != 0)]
    if not len(foreground):
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(foreground, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def crop_bounds(image: np.ndarray, padding: int = 4) -> tuple[slice, slice]:
    foreground = np.isfinite(image) & (np.abs(image) > 1e-6)
    rows, columns = np.where(foreground)
    if not len(rows):
        return slice(0, image.shape[0]), slice(0, image.shape[1])
    return (
        slice(max(int(rows.min()) - padding, 0), min(int(rows.max()) + padding + 1, image.shape[0])),
        slice(max(int(columns.min()) - padding, 0), min(int(columns.max()) + padding + 1, image.shape[1])),
    )


def compact_findings(findings: dict, *, max_fields: int = 5) -> str:
    lines = []
    sites = ", ".join(value.replace("_", " ") for value in findings.get("anatomic_sites", []))
    sides = ", ".join(findings.get("lateralities", []))
    if sites or sides:
        lines.append(f"Location  {sides or '—'} · {sites or '—'}")
    enhancement = findings.get("enhancement_presence")
    if enhancement:
        degree = findings.get("enhancement_degree") or "degree unspecified"
        lines.append(f"Enhancement  {enhancement} · {degree}")
    dimensions = findings.get("largest_reported_dimensions_mm")
    if dimensions:
        lines.append("Size  " + "×".join(f"{value:g}" for value in dimensions) + " mm")
    if findings.get("edema"):
        lines.append(f"Edema  {findings['edema'].replace('_', ' ')}")
    if findings.get("multiplicity"):
        lines.append(f"Multiplicity  {findings['multiplicity']}")
    if findings.get("midline_shift"):
        lines.append(f"Midline shift  {findings['midline_shift']}")
    return "\n".join(lines[:max_fields]) or "No normalized fields extracted"


def compact_deterministic(text: str) -> str:
    patterns = (
        (r"Whole-tumor volume is ([^.]+(?:\.[0-9]+)? mL)\.", "WT volume  {}"),
        (r"Enhancing-tumor volume is ([^.]+(?:\.[0-9]+)? mL)\.", "ET volume  {}"),
        (
            r"Spatial distribution: ([^;]+); ([0-9]+) component\(s\)",
            "Distribution  {} · {} component(s)",
        ),
    )
    lines = []
    for pattern, template in patterns:
        match = re.search(pattern, text)
        if match:
            lines.append(template.format(*match.groups()))
    return "\n".join(lines) or "No deterministic fields rendered"


def wrap_card_text(text: str, width: int) -> str:
    return "\n".join(
        textwrap.fill(line, width=width, subsequent_indent="  ", break_long_words=False)
        for line in text.splitlines()
    )


def text_card(
    axis: plt.Axes,
    text: str,
    *,
    title: str,
    edge: str,
    face: str,
    fontsize: float = 7.2,
    width: int = 34,
) -> None:
    axis.add_patch(
        Rectangle(
            (0.008, 0.018),
            0.984,
            0.964,
            transform=axis.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.8,
        )
    )
    axis.text(
        0.045,
        0.89,
        title.upper(),
        transform=axis.transAxes,
        va="top",
        fontsize=5.7,
        fontweight="bold",
        color=edge,
    )
    axis.text(
        0.045,
        0.69,
        wrap_card_text(text, width),
        transform=axis.transAxes,
        va="top",
        fontsize=fontsize,
        linespacing=1.27,
    )


def main() -> None:
    input_dir = ROOT / "artifacts/private/qualitative_development_cases"
    case_index = json.loads((input_dir / "index.json").read_text(encoding="utf-8"))
    if case_index.get("locked_test_opened") is not False:
        raise RuntimeError("Qualitative images are not explicitly development-only")
    cases = {record["alias"]: record for record in case_index["cases"]}
    records = {
        record["subject_id"]: record
        for record in json.loads(
            (ROOT / "artifacts/private/reporter_records_val.json").read_text(encoding="utf-8")
        )
    }
    paths = {
        "deterministic": "outputs/reporter_deterministic_evidence/val_predictions.json",
        "evidence": "outputs/llava_med_v15_mistral_7b_evidence_only/val_predictions_unconstrained.json",
        "full": "outputs/llava_med_v15_mistral_7b_full/val_predictions_unconstrained.json",
    }
    predictions = {
        name: {
            record["subject_id"]: record
            for record in json.loads((ROOT / path).read_text(encoding="utf-8"))
        }
        for name, path in paths.items()
    }
    selected = [cases[alias] for alias in ALIASES]
    if any(record["subject_id"] not in records for record in selected):
        raise ValueError("Prespecified Figure 5 cases are not all in report validation")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": COLORS["INK"]})
    figure, axes = plt.subplots(
        3,
        5,
        figsize=(11.9, 4.8),
        gridspec_kw={"width_ratios": [0.93, 1.14, 1.03, 1.28, 1.28]},
    )
    figure.subplots_adjust(
        left=0.012,
        right=0.997,
        top=0.905,
        bottom=0.052,
        wspace=0.012,
        hspace=0.012,
    )
    for row, case in enumerate(selected):
        subject_id = case["subject_id"]
        record = records[subject_id]
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

        image_axis = axes[row, 0]
        image_axis.imshow(base, cmap="gray", vmin=0, vmax=1)
        if target_wt.any():
            image_axis.contour(target_wt, levels=[0.5], colors=[COLORS["GT"]], linewidths=1.5)
        if predicted_wt.any():
            image_axis.contour(
                predicted_wt,
                levels=[0.5],
                colors=[COLORS["PRED"]],
                linewidths=1.4,
                linestyles="--",
            )
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        for spine in image_axis.spines.values():
            spine.set_visible(False)
        image_axis.text(
            0.03,
            0.97,
            "FLAIR",
            transform=image_axis.transAxes,
            color="white",
            va="top",
            fontsize=7,
            fontweight="bold",
            bbox={"facecolor": COLORS["INK"], "alpha": 0.72, "pad": 1.2, "edgecolor": "none"},
        )
        image_axis.text(
            0.97,
            0.97,
            case["alias"],
            transform=image_axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="white",
            bbox={"facecolor": COLORS["INK"], "alpha": 0.72, "pad": 1.2, "edgecolor": "none"},
        )

        reference_axis = axes[row, 1]
        deterministic_axis = axes[row, 2]
        evidence_axis = axes[row, 3]
        full_axis = axes[row, 4]
        for axis in (reference_axis, deterministic_axis, evidence_axis, full_axis):
            axis.axis("off")

        text_card(
            reference_axis,
            compact_findings(record["normalized_findings"]),
            title="Reference · rule-extracted",
            edge=COLORS["REFERENCE"],
            face="#F5F7FA",
            fontsize=5.9,
            width=27,
        )
        deterministic = predictions["deterministic"][subject_id]
        text_card(
            deterministic_axis,
            compact_deterministic(deterministic["prediction"]),
            title="Family field suppressed",
            edge=COLORS["SAFE"],
            face="#EFF8F8",
            fontsize=5.9,
            width=25,
        )

        evidence = predictions["evidence"][subject_id]
        full = predictions["full"][subject_id]
        evidence_title = (
            f"Unsupported {evidence['structured_field_unsupported_rate']:.0%} · "
            f"contradiction {evidence['structured_field_contradiction_rate']:.0%}"
        )
        full_title = (
            f"Unsupported {full['structured_field_unsupported_rate']:.0%} · "
            f"contradiction {full['structured_field_contradiction_rate']:.0%}"
        )
        text_card(
            evidence_axis,
            compact_findings(evidence["predicted_normalized_findings"], max_fields=4),
            title=evidence_title,
            edge=COLORS["CAUTION"],
            face="#F1F5FB",
            fontsize=5.65,
            width=27,
        )
        text_card(
            full_axis,
            compact_findings(full["predicted_normalized_findings"], max_fields=4),
            title=full_title,
            edge=COLORS["FAIL"],
            face="#F6F2FA",
            fontsize=5.65,
            width=27,
        )

    for axis, title in zip(
        axes[0],
        (
            "MRI + WT contours",
            "Reference facts",
            "Traceable renderer",
            "LLaVA-Med · evidence only",
            "LLaVA-Med · visual + evidence",
        ),
        strict=True,
    ):
        axis.set_title(title, fontweight="bold", fontsize=8.0, pad=7)
    figure.legend(
        handles=(
            Line2D([0], [0], color=COLORS["GT"], lw=2, label="Reference WT"),
            Line2D([0], [0], color=COLORS["PRED"], lw=2, ls="--", label="Predicted WT"),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=1.8,
        columnspacing=1.2,
    )
    output = ROOT / "Manuscript/Figures/figure5_qualitative_report_comparison"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    public = {
        "scope": "development report-validation only; pending clinical review",
        "selection": "cases frozen for Figure 3 before reporter-output inspection; one per cohort with report-validation overlap",
        "locked_test_opened": False,
        "cases": [
            {"alias": case["alias"], "cohort": case["cohort"], "role": case["role"]}
            for case in selected
        ],
        "prediction_sources": paths,
        "display": "normalized structured fields; source-limited family output suppressed; global per-case unsupported and contradiction rates; compact gutters",
    }
    output.with_suffix(".json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
