#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import MODALITIES
from tumortrust_vlm.data.preprocessing import load_preprocessed_case
from tumortrust_vlm.utils import atomic_json_dump


def display_slice(array: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(array[:, :, index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--output", default="artifacts/private/qc/preprocessing_montage.png"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    inventory = json.loads(
        (ROOT / config["data"]["inventory"]).read_text(encoding="utf-8")
    )
    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in inventory:
        if record.get("eligible_for_split"):
            key = (record["cohort"], record["source_branch"], record["orientation"])
            strata[key].append(record)
    selected = [strata[key][0] for key in sorted(strata)]
    figure, axes = plt.subplots(
        len(selected), len(MODALITIES) + 1, figsize=(15, 3 * len(selected)), squeeze=False
    )
    summaries = []
    for row, record in enumerate(selected):
        image, mask, _, _ = load_preprocessed_case(
            record,
            target_spacing=config["data"]["target_spacing"],
            normalization=config["data"]["normalization"],
            crop_margin=config["data"]["brain_crop_margin"],
            require_segmentation=True,
        )
        assert mask is not None
        slice_index = int(np.argmax((mask > 0).sum(axis=(0, 1))))
        for column, modality in enumerate(MODALITIES):
            axes[row, column].imshow(display_slice(image[column], slice_index), cmap="gray")
            axes[row, column].set_title(modality)
            axes[row, column].axis("off")
        axes[row, -1].imshow(display_slice(image[1], slice_index), cmap="gray")
        overlay = np.ma.masked_where(display_slice(mask, slice_index) == 0, display_slice(mask, slice_index))
        axes[row, -1].imshow(overlay, cmap="turbo", alpha=0.45, interpolation="nearest")
        axes[row, -1].set_title("T1c + label")
        axes[row, -1].axis("off")
        axes[row, 0].set_ylabel(" / ".join((record["cohort"], record["source_branch"], record["orientation"])))
        summaries.append(
            {
                "subject_id": record["subject_id"],
                "cohort": record["cohort"],
                "source_branch": record["source_branch"],
                "orientation": record["orientation"],
                "slice_index": slice_index,
                "label_values": [int(value) for value in np.unique(mask)],
            }
        )
    figure.suptitle("TumorTrust-VLM canonical preprocessing QC", fontsize=14)
    figure.tight_layout()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    atomic_json_dump(summaries, output.with_suffix(".json"))
    atomic_json_dump(
        {"strata_rendered": len(selected), "montage_generated": True},
        ROOT / "artifacts" / "visual_qc_summary.json",
    )
    print(json.dumps({"strata": len(selected), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
