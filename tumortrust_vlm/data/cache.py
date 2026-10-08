from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tumortrust_vlm.data.preprocessing import load_preprocessed_case
from tumortrust_vlm.utils import atomic_json_dump


def preprocessing_key(config: dict) -> str:
    selected = {
        "target_spacing": config["data"].get("target_spacing", [1, 1, 1]),
        "normalization": config["data"].get("normalization", "robust_nonzero"),
        "brain_crop_margin": config["data"].get("brain_crop_margin", 8),
        "canonicalize": config["data"].get("canonicalize", True),
        "version": 1,
    }
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16]


def cache_path(cache_root: str | Path, key: str, subject_id: str) -> Path:
    return Path(cache_root) / key / f"{subject_id}.npz"


def cache_summary_path(cache_root: str | Path, key: str) -> Path:
    return Path(cache_root).parent / "cache_summaries" / f"{key}.json"


def _cache_one(arguments: tuple[dict, dict, str, str]) -> dict:
    record, data_config, root, key = arguments
    destination = cache_path(root, key, record["subject_id"])
    if destination.is_file():
        return {"subject_id": record["subject_id"], "status": "existing", "bytes": destination.stat().st_size}
    image, label, transform, presence = load_preprocessed_case(
        record,
        target_spacing=data_config.get("target_spacing", (1, 1, 1)),
        normalization=data_config.get("normalization", "robust_nonzero"),
        crop_margin=data_config.get("brain_crop_margin", 8),
        require_segmentation=True,
        canonicalize=data_config.get("canonicalize", True),
    )
    assert label is not None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            image=image.astype(np.float16),
            label=label.astype(np.uint8),
            spacing=np.asarray(transform.spacing, dtype=np.float32),
            modality_presence=presence,
        )
    os.replace(temporary, destination)
    return {"subject_id": record["subject_id"], "status": "written", "bytes": destination.stat().st_size}


def build_cache(records: list[dict], config: dict, workers: int = 4) -> dict:
    key = preprocessing_key(config)
    root = config["data"]["cache_dir"]
    eligible = [record for record in records if record.get("eligible_for_split", True) and record["labeled"]]
    arguments = [(record, config["data"], root, key) for record in eligible]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_cache_one, argument) for argument in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 100 == 0:
                print(f"[cache] {completed}/{len(futures)}", flush=True)
    summary = {
        "preprocessing_key": key,
        "subjects": len(results),
        "written": sum(result["status"] == "written" for result in results),
        "existing": sum(result["status"] == "existing" for result in results),
        "total_bytes": sum(result["bytes"] for result in results),
    }
    atomic_json_dump(summary, cache_summary_path(root, key))
    return summary


def load_cached_case(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        return (
            payload["image"].astype(np.float32),
            payload["label"].astype(np.int16),
            payload["spacing"].astype(np.float32),
            payload["modality_presence"].astype(np.float32),
        )
