#!/usr/bin/env python
"""Render the outcome-aware, four-block TumorTrust manuscript framework."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]


def rounded(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    linestyle: str = "-",
    linewidth: float = 1.6,
    radius: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=axis.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    axis.add_patch(patch)
    return patch


def block_title(axis: plt.Axes, x: float, number: int, title: str, color: str) -> None:
    axis.text(
        x + 0.025,
        0.79,
        str(number),
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "circle,pad=0.35", "facecolor": color, "edgecolor": "none"},
    )
    axis.text(
        x + 0.055,
        0.79,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )


def arrow(axis: plt.Axes, start: float, end: float, label: str, color: str) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (start, 0.49),
            (end, 0.49),
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color=color,
        )
    )
    axis.text(
        (start + end) / 2,
        0.53,
        label,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axis = plt.subplots(figsize=(14, 5.1))
    axis.set_axis_off()
    figure.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.06)

    starts = (0.015, 0.265, 0.515, 0.765)
    faces = ("#F4F6F8", "#E7F7F3", "#FFF5D9", "#EEF4FF")
    edges = ("#657185", "#008F80", "#B77900", "#3569C8")
    titles = (
        "Segmentation comparison",
        "D3 deterministic evidence",
        "D3 trust audit",
        "Reporting decision",
    )
    for x, face, edge in zip(starts, faces, edges, strict=True):
        rounded(axis, (x, 0.16), 0.22, 0.67, face=face, edge=edge, linewidth=2)
    for index, (x, title, color) in enumerate(zip(starts, titles, edges, strict=True), start=1):
        block_title(axis, x, index, title, color)

    axis.text(
        starts[0] + 0.11,
        0.58,
        "T1 · T1c · T2 · FLAIR\nregistered, canonical 3D volumes",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        linespacing=1.4,
    )
    rounded(
        axis,
        (starts[0] + 0.035, 0.34),
        0.15,
        0.13,
        face="white",
        edge="#657185",
        radius=0.015,
    )
    axis.text(
        starts[0] + 0.11,
        0.405,
        "nnU-Net benchmark\nSegResNet D3 downstream source",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=8.3,
    )
    axis.text(
        starts[0] + 0.11,
        0.255,
        "Same subjects and label ontology\ndifferent fitted preprocessing and training\nstandard CNNs, not architectural novelty",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#4B5563",
        linespacing=1.35,
    )

    axis.text(
        starts[1] + 0.11,
        0.60,
        "D3 predicted mask → measured findings",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
    )
    rounded(
        axis,
        (starts[1] + 0.03, 0.35),
        0.16,
        0.16,
        face="#C8F3EA",
        edge="#008F80",
        radius=0.015,
    )
    axis.text(
        starts[1] + 0.11,
        0.43,
        "Volumes (mL) · laterality\ncomponents · region-aware tokens",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.7,
    )
    axis.text(
        starts[1] + 0.11,
        0.255,
        "Structured evidence card\nexact value trace · explicit unavailable fields\nmask-derived burden; no free regression",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="#335C57",
        linespacing=1.35,
    )

    axis.text(
        starts[2] + 0.11,
        0.60,
        "D3 calibration · intervals · uncertainty",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
    )
    rounded(
        axis,
        (starts[2] + 0.03, 0.35),
        0.16,
        0.16,
        face="#FFEAB5",
        edge="#B77900",
        radius=0.015,
    )
    axis.text(
        starts[2] + 0.11,
        0.43,
        "Cohort / source LODO\nmissing sequence · controlled shift",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.7,
    )
    axis.text(
        starts[2] + 0.11,
        0.255,
        "Measure failure before referral\nall tested D3 trust gates failed\nno nnU-Net referral claim",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="#6D5200",
        linespacing=1.35,
    )

    rounded(
        axis,
        (starts[3] + 0.025, 0.49),
        0.17,
        0.16,
        face="#DDF5EC",
        edge="#008F80",
        radius=0.015,
    )
    axis.text(
        starts[3] + 0.11,
        0.57,
        "RETAINED\nDeterministic, traceable report\nomit · qualify · specialist review",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        linespacing=1.3,
    )
    rounded(
        axis,
        (starts[3] + 0.025, 0.25),
        0.17,
        0.16,
        face="#FCE8E6",
        edge="#C43C35",
        linestyle="--",
        radius=0.015,
    )
    axis.text(
        starts[3] + 0.11,
        0.33,
        "REJECTED ON DEVELOPMENT\nLearned visual-language reporter\nfailed factual + evidence-swap gates",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#8F2924",
        linespacing=1.3,
    )

    arrow(axis, 0.232, 0.266, "D3 mask", "#657185")
    arrow(axis, 0.482, 0.516, "evidence", "#008F80")
    arrow(axis, 0.732, 0.766, "audited evidence", "#B77900")

    footers = (
        "NNUNET BENCHMARK / D3 AUDIT SOURCE",
        "D3 STRUCTURED EVIDENCE LAYER",
        "D3 TRUST GATES FAILED; REFERRAL DISABLED",
        "OUTCOME-GATED REPORTING",
    )
    for x, text, color in zip(starts, footers, edges, strict=True):
        axis.text(
            x + 0.11,
            0.105,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=color,
        )

    output = ROOT / "Manuscript/Figures/figure1_locked_framework_manuscript"
    crop = Bbox.from_extents(0.10, 0.65, 13.90, 4.20)
    figure.savefig(output.with_suffix(".svg"), bbox_inches=crop, pad_inches=0)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches=crop, pad_inches=0)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches=crop, pad_inches=0)
    plt.close(figure)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
