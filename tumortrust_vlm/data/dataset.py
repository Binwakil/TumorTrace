from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tumortrust_vlm.data.cache import cache_path, load_cached_case, preprocessing_key
from tumortrust_vlm.data.constants import COHORT_TO_INDEX
from tumortrust_vlm.data.preprocessing import (
    apply_input_view,
    apply_robustness_shift,
    extract_patch,
    load_preprocessed_case,
    simulate_missing_modalities,
)


class TumorTrustDataset(Dataset):
    def __init__(
        self,
        inventory_path: str | Path,
        split_manifest_path: str | Path,
        *,
        split: str,
        patch_size: Sequence[int] | None = None,
        training: bool = False,
        seed: int = 0,
        normalization: str = "robust_nonzero",
        target_spacing: Sequence[float] = (1, 1, 1),
        crop_margin: int | None = 8,
        positive_probability: float = 0.75,
        canonicalize: bool = True,
        input_view: str = "whole",
        missing_modality_probability: float = 0.0,
        force_missing_modality: str | None = None,
        subject_ids: set[str] | None = None,
        cache_dir: str | Path | None = None,
        cache_config: dict | None = None,
        label_mode: str = "cohort_native_4class",
        robustness_shift: str = "none",
    ) -> None:
        inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
        by_id = {record["subject_id"]: record for record in inventory}
        selected = [entry for entry in manifest["entries"] if entry["split"] == split]
        if subject_ids is not None:
            selected = [entry for entry in selected if entry["subject_id"] in subject_ids]
        self.records = [by_id[entry["subject_id"]] for entry in selected]
        self.patch_size = tuple(patch_size) if patch_size else None
        self.training = training
        self.seed = seed
        self.normalization = normalization
        self.target_spacing = target_spacing
        self.crop_margin = crop_margin
        self.positive_probability = positive_probability
        self.canonicalize = canonicalize
        self.input_view = input_view
        self.missing_modality_probability = missing_modality_probability
        self.force_missing_modality = force_missing_modality
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_key = preprocessing_key(cache_config) if cache_dir and cache_config else None
        self.label_mode = label_mode
        self.robustness_shift = robustness_shift
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible but different augmentation stream for each epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        cached = (
            cache_path(self.cache_dir, self.cache_key, record["subject_id"])
            if self.cache_dir is not None and self.cache_key is not None
            else None
        )
        if cached is not None and cached.is_file():
            image, segmentation, spacing, _ = load_cached_case(cached)
            transform = None
        else:
            image, segmentation, transform, _ = load_preprocessed_case(
                record,
                target_spacing=self.target_spacing,
                normalization=self.normalization,
                crop_margin=self.crop_margin,
                require_segmentation=True,
                canonicalize=self.canonicalize,
            )
            spacing = np.asarray(transform.spacing, dtype=np.float32)
        assert segmentation is not None
        if self.label_mode == "whole_tumor":
            segmentation = (segmentation > 0).astype(np.int16)
        elif self.label_mode != "cohort_native_4class":
            raise ValueError(f"Unknown label mode: {self.label_mode}")
        image = apply_input_view(image, segmentation, self.input_view)
        image = apply_robustness_shift(image, self.robustness_shift, rng)
        image, presence = simulate_missing_modalities(
            image,
            rng,
            probability=self.missing_modality_probability if self.training else 0.0,
            force=self.force_missing_modality,
        )
        if self.patch_size is not None:
            image, segmentation = extract_patch(
                image,
                segmentation,
                self.patch_size,
                rng,
                self.positive_probability if self.training else 1.0,
            )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(segmentation)).long(),
            "class_label": torch.tensor(COHORT_TO_INDEX[record["cohort"]], dtype=torch.long),
            "modality_presence": torch.from_numpy(presence),
            "subject_id": record["subject_id"],
            "cohort": record["cohort"],
            "source_branch": record["source_branch"],
            "source_orientation": record.get("orientation", "unknown"),
            "has_report": bool(record.get("has_report", False)),
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "transform": transform,
        }


def collate_training(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "class_label": torch.stack([item["class_label"] for item in batch]),
        "modality_presence": torch.stack([item["modality_presence"] for item in batch]),
        "subject_id": [item["subject_id"] for item in batch],
        "cohort": [item["cohort"] for item in batch],
        "source_branch": [item["source_branch"] for item in batch],
        "source_orientation": [item["source_orientation"] for item in batch],
        "has_report": [item["has_report"] for item in batch],
        "spacing": torch.stack([item["spacing"] for item in batch]),
        "transform": [item["transform"] for item in batch],
    }
