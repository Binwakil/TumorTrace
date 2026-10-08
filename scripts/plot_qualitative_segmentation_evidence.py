#!/usr/bin/env python
"""Render manuscript Figure 3 from prespecified development-only exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "WT": "#009E73",
    "TC": "#E69F00",
    "ET": "#D55E00",
    "SNFH": "#0072B2",
    "FP": "#CC79A7",
    "FN": "#56B4E9",
    "INK": "#172033",
    "MUTED": "#667085",
}
REGIONS = {"WT": (1, 2, 3, 4), "TC": (1, 3, 4), "ET": (3, 4), "SNFH": (2,)}


def display_slice(volume: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(volume[:, :, index])


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
    row_min = max(int(rows.min()) - padding, 0)
    row_max = min(int(rows.max()) + padding + 1, image.shape[0])
    column_min = max(int(columns.min()) - padding, 0)
    column_max = min(int(columns.max()) + padding + 1, image.shape[1])
    size = min(max(row_max - row_min, column_max - column_min), min(image.shape))

    def centered_window(start: int, stop: int, limit: int) -> slice:
        center = (start + stop) / 2
        new_start = max(0, min(round(center - size / 2), limit - size))
        return slice(new_start, new_start + size)

    return (
        centered_window(row_min, row_max, image.shape[0]),
        centered_window(column_min, column_max, image.shape[1]),
    )


def contours(axis: plt.Axes, mask: np.ndarray, *, linestyle: str = "-") -> None:
    for region, labels in REGIONS.items():
        binary = np.isin(mask, labels)
        if binary.any() and not binary.all():
            axis.contour(
                binary,
                levels=[0.5],
                colors=[COLORS[region]],
                linewidths=1.25,
                linestyles=linestyle,
            )


def clean_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/private/qualitative_development_cases")
    parser.add_argument(
        "--output-stem",
        default="Manuscript/Figures/figure3_qualitative_segmentation_evidence",
    )
    parser.add_argument("--role", default="typical")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    index = json.loads((input_dir / "index.json").read_text(encoding="utf-8"))
    if index.get("locked_test_opened") is not False:
        raise RuntimeError("Qualitative input is not explicitly development-only")
    cases = [record for record in index["cases"] if record["role"] == args.role]
    if [record["cohort"] for record in cases] != ["GLI", "MEN", "MET"]:
        raise ValueError("Figure requires one prespecified GLI/MEN/MET case")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlecolor": COLORS["INK"],
            "text.color": COLORS["INK"],
        }
    )
    figure = plt.figure(figsize=(7.6, 10.7), facecolor="white")
    grid = figure.add_gridspec(
        5,
        3,
        height_ratios=(1.0, 1.0, 1.0, 1.0, 0.32),
        left=0.088,
        right=0.995,
        top=0.925,
        bottom=0.075,
        wspace=0.006,
        hspace=0.008,
    )
    row_anchor_axes: list[plt.Axes] = []
    for column, record in enumerate(cases):
        data = np.load(input_dir / record["data_file"])
        image = data["image"].astype(np.float32)
        target = np.squeeze(data["target"]).astype(np.uint8)
        prediction = data["prediction"].astype(np.uint8)
        slice_index = int(np.isin(target, REGIONS["WT"]).sum(axis=(0, 1)).argmax())

        displayed = [normalize(display_slice(image[index], slice_index)) for index in range(4)]
        bounds = crop_bounds(displayed[3])
        displayed = [sequence[bounds] for sequence in displayed]
        target_slice = display_slice(target, slice_index)[bounds]
        prediction_slice = display_slice(prediction, slice_index)[bounds]
        base = displayed[3]

        sequence_grid = grid[0, column].subgridspec(2, 2, wspace=0.012, hspace=0.012)
        sequence_axes = []
        for sequence_index, label in enumerate(("T1", "T1c", "T2", "FLAIR")):
            axis = figure.add_subplot(sequence_grid[sequence_index // 2, sequence_index % 2])
            axis.imshow(displayed[sequence_index], cmap="gray", vmin=0, vmax=1)
            axis.text(
                0.05,
                0.94,
                label,
                transform=axis.transAxes,
                color="white",
                va="top",
                fontsize=7.2,
                fontweight="bold",
                bbox={"facecolor": "#172033", "alpha": 0.72, "pad": 1.2, "edgecolor": "none"},
            )
            clean_image_axis(axis)
            sequence_axes.append(axis)

        ground_truth_axis = figure.add_subplot(grid[1, column])
        prediction_axis = figure.add_subplot(grid[2, column])
        error_axis = figure.add_subplot(grid[3, column])
        evidence_axis = figure.add_subplot(grid[4, column])
        overlay_axes = (ground_truth_axis, prediction_axis, error_axis)
        for axis in overlay_axes:
            axis.imshow(base, cmap="gray", vmin=0, vmax=1)
            clean_image_axis(axis)
        contours(ground_truth_axis, target_slice)
        contours(prediction_axis, prediction_slice)

        target_wt = np.isin(target_slice, REGIONS["WT"])
        predicted_wt = np.isin(prediction_slice, REGIONS["WT"])
        false_positive = predicted_wt & ~target_wt
        false_negative = target_wt & ~predicted_wt
        if false_positive.any():
            error_axis.contourf(false_positive, levels=[0.5, 1.5], colors=[COLORS["FP"]], alpha=0.78)
        if false_negative.any():
            error_axis.contourf(false_negative, levels=[0.5, 1.5], colors=[COLORS["FN"]], alpha=0.78)
        if target_wt.any():
            error_axis.contour(target_wt, levels=[0.5], colors=["white"], linewidths=0.65, alpha=0.75)

        evidence = record["evidence"]
        evidence_axis.axis("off")
        evidence_axis.text(
            0.035,
            0.76,
            (
                f"WT {evidence['volumes']['WT']['value_ml']:.1f} mL  ·  "
                f"{evidence['laterality']}  ·  {evidence['component_count']} component(s)\n"
                f"Dice {record['macro_dice']:.3f}  ·  entropy p95 "
                f"{evidence['segmentation_uncertainty']:.3f}\n"
                "Family output suppressed"
            ),
            transform=evidence_axis.transAxes,
            va="top",
            fontsize=6.35,
            linespacing=1.25,
        )
        evidence_axis.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=evidence_axis.transAxes,
                facecolor="#F4F6F8",
                edgecolor="#D0D5DD",
                linewidth=0.75,
                zorder=-1,
                clip_on=False,
            )
        )

        position = grid[0, column].get_position(figure)
        figure.text(
            (position.x0 + position.x1) / 2,
            0.94,
            f"{record['alias']}  ·  {record['cohort']}  ·  Dice {record['macro_dice']:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
        )
        if column == 0:
            row_anchor_axes = [sequence_axes[0], ground_truth_axis, prediction_axis, error_axis, evidence_axis]

    row_titles = ("Four-sequence MRI", "Reference", "Prediction", "WT error", "Evidence")
    for row, (_axis, title) in enumerate(zip(row_anchor_axes, row_titles, strict=True)):
        position = grid[row, 0].get_position(figure)
        figure.text(
            0.022,
            (position.y0 + position.y1) / 2,
            title,
            ha="center",
            va="center",
            rotation=90,
            fontsize=7.5,
            fontweight="bold",
        )

    legend = [
        Line2D([0], [0], color=COLORS[region], lw=2, label=region)
        for region in ("WT", "TC", "ET", "SNFH")
    ] + [
        Patch(facecolor=COLORS["FP"], label="False positive"),
        Patch(facecolor=COLORS["FN"], label="False negative"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.018),
        ncol=6,
        frameon=False,
        fontsize=7.1,
        handlelength=1.6,
        columnspacing=1.0,
    )
    output = ROOT / args.output_stem
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    public_index = {
        "scope": "development-only; no locked final-test data",
        "selection_policy": "qualitative_policy_v1",
        "role": args.role,
        "cases": [
            {
                "alias": record["alias"],
                "cohort": record["cohort"],
                "role": record["role"],
                "macro_dice": record["macro_dice"],
            }
            for record in cases
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(public_index, indent=2) + "\n", encoding="utf-8"
    )
    print(output.with_suffix(".pdf"))
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
