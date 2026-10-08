#!/usr/bin/env python
"""Render development-only overlay and error galleries from frozen case exports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts/private/qualitative_development_cases"
OUTPUT_DIR = ROOT / "Manuscript/Figures"
COLORS = {
    "INK": "#172033",
    "MUTED": "#667085",
    "GT": "#009E73",
    "PRED": "#E69F00",
    "FP": "#CC79A7",
    "FN": "#56B4E9",
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


def load_slice(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(INPUT_DIR / record["data_file"])
    image = data["image"].astype(np.float32)
    target = np.squeeze(data["target"]).astype(np.uint8)
    prediction = data["prediction"].astype(np.uint8)
    slice_index = int((target > 0).sum(axis=(0, 1)).argmax())
    base = np.rot90(normalize(image[3, :, :, slice_index]))
    target_wt = np.rot90(target[:, :, slice_index]) > 0
    predicted_wt = np.rot90(prediction[:, :, slice_index]) > 0
    bounds = crop_bounds(base)
    return base[bounds], target_wt[bounds], predicted_wt[bounds]


def clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_figure(figure: plt.Figure, stem: str) -> None:
    output = OUTPUT_DIR / stem
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def overlay_gallery(cases: list[dict]) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(8.6, 8.2))
    figure.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.995,
        bottom=0.075,
        wspace=0.008,
        hspace=0.008,
    )
    for axis, record in zip(axes.flat, cases, strict=True):
        base, target_wt, predicted_wt = load_slice(record)
        axis.imshow(base, cmap="gray", vmin=0, vmax=1)
        if target_wt.any():
            axis.contour(target_wt, levels=[0.5], colors=[COLORS["GT"]], linewidths=1.7)
        if predicted_wt.any():
            axis.contour(
                predicted_wt,
                levels=[0.5],
                colors=[COLORS["PRED"]],
                linewidths=1.5,
                linestyles="--",
            )
        clean_axis(axis)
        axis.text(
            0.025,
            0.975,
            f"{record['alias']} · {record['role'].replace('_', ' ')}\n"
            f"WT {record['wt_dice']:.3f} · macro {record['macro_dice']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            fontweight="bold",
            color="white",
            bbox={"facecolor": COLORS["INK"], "alpha": 0.76, "pad": 1.4, "edgecolor": "none"},
        )
    for row, cohort in enumerate(("GLI", "MEN", "MET")):
        axes[row, 0].text(
            -0.095,
            0.5,
            cohort,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    figure.legend(
        handles=(
            Line2D([0], [0], color=COLORS["GT"], lw=2, label="Reference WT"),
            Line2D([0], [0], color=COLORS["PRED"], lw=2, ls="--", label="Predicted WT"),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        frameon=False,
    )
    save_figure(figure, "figureS1_development_overlay_gallery")


def error_gallery(cases: list[dict]) -> None:
    selected = [record for record in cases if record["role"] != "typical"]
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 5.75))
    figure.subplots_adjust(
        left=0.01,
        right=0.995,
        top=0.995,
        bottom=0.105,
        wspace=0.008,
        hspace=0.008,
    )
    for axis, record in zip(axes.flat, selected, strict=True):
        base, target_wt, predicted_wt = load_slice(record)
        false_positive = predicted_wt & ~target_wt
        false_negative = target_wt & ~predicted_wt
        axis.imshow(base, cmap="gray", vmin=0, vmax=1)
        if false_positive.any():
            axis.contourf(false_positive, levels=[0.5, 1.5], colors=[COLORS["FP"]], alpha=0.82)
        if false_negative.any():
            axis.contourf(false_negative, levels=[0.5, 1.5], colors=[COLORS["FN"]], alpha=0.82)
        if target_wt.any():
            axis.contour(target_wt, levels=[0.5], colors=["white"], linewidths=0.8)
        clean_axis(axis)
        axis.text(
            0.025,
            0.975,
            f"{record['alias']} · {record['cohort']} · {record['role'].replace('_', ' ')}\n"
            f"WT Dice {record['wt_dice']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color="white",
            bbox={"facecolor": COLORS["INK"], "alpha": 0.76, "pad": 1.4, "edgecolor": "none"},
        )
    figure.legend(
        handles=(
            Patch(facecolor=COLORS["FP"], label="False positive"),
            Patch(facecolor=COLORS["FN"], label="False negative"),
            Line2D([0], [0], color="white", markeredgecolor=COLORS["MUTED"], lw=2, label="Reference WT boundary"),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )
    save_figure(figure, "figureS2_development_error_gallery")


def main() -> None:
    index = json.loads((INPUT_DIR / "index.json").read_text(encoding="utf-8"))
    if index.get("locked_test_opened") is not False:
        raise RuntimeError("Qualitative input is not explicitly development-only")
    cases = index["cases"]
    if len(cases) != 9:
        raise ValueError("Expected the frozen nine-case qualitative set")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_gallery(cases)
    error_gallery(cases)
    common = {
        "scope": "development-only; locked final test unopened",
        "locked_test_opened": False,
        "selection_policy": "qualitative_policy_v1",
        "source_index": "artifacts/private/qualitative_development_cases/index.json",
    }
    (OUTPUT_DIR / "figureS1_development_overlay_gallery.json").write_text(
        json.dumps(
            {
                **common,
                "cases": [
                    {
                        "alias": record["alias"],
                        "cohort": record["cohort"],
                        "role": record["role"],
                        "wt_dice": record["wt_dice"],
                        "macro_dice": record["macro_dice"],
                    }
                    for record in cases
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    edge_cases = [record for record in cases if record["role"] != "typical"]
    (OUTPUT_DIR / "figureS2_development_error_gallery.json").write_text(
        json.dumps(
            {
                **common,
                "cases": [
                    {
                        "alias": record["alias"],
                        "cohort": record["cohort"],
                        "role": record["role"],
                        "wt_dice": record["wt_dice"],
                    }
                    for record in edge_cases
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR / "figureS1_development_overlay_gallery.pdf")
    print(OUTPUT_DIR / "figureS2_development_error_gallery.pdf")


if __name__ == "__main__":
    main()
