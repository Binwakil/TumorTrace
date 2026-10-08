from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable

from tumortrust_vlm.utils import sha256_json


def _allocate(
    items: list[dict], rng: random.Random, val_fraction: float, test_fraction: float
) -> None:
    rng.shuffle(items)
    count = len(items)
    n_test = round(count * test_fraction)
    n_val = round(count * val_fraction)
    for index, record in enumerate(items):
        if index < n_test:
            record["split"] = "test"
        elif index < n_test + n_val:
            record["split"] = "val"
        else:
            record["split"] = "train"
        record["split_reason"] = "source_stratified_nonreport"


def build_master_split(
    records: Iterable[dict],
    *,
    seed: int,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> dict:
    eligible = [
        dict(record)
        for record in records
        if record["labeled"]
        and record["complete_images"]
        and record.get("eligible_for_split", True)
    ]
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in eligible:
        if record.get("report_split"):
            record["split"] = record["report_split"]
            record["split_reason"] = "locked_report_split"
        else:
            by_stratum[(record["cohort"], record["source_branch"])].append(record)
    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        _allocate(by_stratum[stratum], rng, val_fraction, test_fraction)

    fingerprint_splits: dict[str, set[str]] = defaultdict(set)
    for record in eligible:
        fingerprint = record.get("image_fingerprint") or record["identity_fingerprint"]
        fingerprint_splits[fingerprint].add(record["split"])
    conflicts = [
        fingerprint for fingerprint, values in fingerprint_splits.items() if len(values) > 1
    ]
    if conflicts:
        raise ValueError(f"Duplicate content crosses splits ({len(conflicts)} groups)")

    entries = [
        {
            "subject_id": record["subject_id"],
            "cohort": record["cohort"],
            "source_branch": record["source_branch"],
            "split": record["split"],
            "split_reason": record["split_reason"],
            "has_report": record["has_report"],
        }
        for record in sorted(eligible, key=lambda item: (item["cohort"], item["subject_id"]))
    ]
    lodo = {
        held_out: {
            "train": [
                entry["subject_id"]
                for entry in entries
                if entry["cohort"] != held_out and entry["split"] == "train"
            ],
            "val": [
                entry["subject_id"]
                for entry in entries
                if entry["cohort"] != held_out and entry["split"] == "val"
            ],
            "test": [entry["subject_id"] for entry in entries if entry["cohort"] == held_out],
            "task": "common_WT_segmentation_and_burden_only",
        }
        for held_out in ("GLI", "MEN", "MET")
    }
    met_branches = sorted({entry["source_branch"] for entry in entries if entry["cohort"] == "MET"})
    loso_met = {
        branch: {
            "train": [
                entry["subject_id"]
                for entry in entries
                if entry["cohort"] == "MET" and entry["source_branch"] != branch
            ],
            "test": [
                entry["subject_id"]
                for entry in entries
                if entry["cohort"] == "MET" and entry["source_branch"] == branch
            ],
        }
        for branch in met_branches
    }
    manifest = {
        "seed": seed,
        "entries": entries,
        "counts": dict(Counter(entry["split"] for entry in entries)),
        "counts_by_cohort_split": {
            f"{cohort}:{split}": sum(
                entry["cohort"] == cohort and entry["split"] == split for entry in entries
            )
            for cohort in ("GLI", "MEN", "MET")
            for split in ("train", "val", "test")
        },
        "lodo": lodo,
        "loso_met": loso_met,
        "final_test_locked": True,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def assert_zero_overlap(manifest: dict) -> None:
    split_sets = {
        split: {entry["subject_id"] for entry in manifest["entries"] if entry["split"] == split}
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise AssertionError(f"{left}/{right} overlap: {sorted(overlap)[:5]}")


def split_requires_final_unlock(manifest: dict, split: str) -> bool:
    """Return whether evaluating a role exposes subjects from the frozen final test.

    Cross-evaluation manifests use ``test`` as the held-out evaluation *role* even
    during development, where those subjects originate from the master validation
    split.  New manifests record that provenance explicitly.  The conservative
    fallback preserves the lock for older/master manifests.
    """
    if split != "test":
        return False
    if "contains_source_final_test" in manifest:
        return bool(manifest["contains_source_final_test"])
    return bool(manifest.get("final_test_locked", True))
