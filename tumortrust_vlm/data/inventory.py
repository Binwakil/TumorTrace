from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

from tumortrust_vlm.data.constants import (
    COHORT_DIRECTORIES,
    COHORTS,
    CORRECTION_BRANCH,
    LABELED_BRANCHES,
    MODALITIES,
    REPORT_DIRECTORIES,
    UNLABELED_BRANCHES,
)


def _resolve_file(folder: Path, subject_id: str, suffix: str) -> Path | None:
    candidates = [
        folder / f"{subject_id}-{suffix}{extension}" for extension in (".nii.gz", ".nii")
    ]
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if len(candidates) == 1:
        return candidates[0].resolve()
    for candidate in candidates:
        try:
            np.asarray(nib.load(candidate).dataobj)
            return candidate.resolve()
        except (OSError, EOFError, ValueError, zlib.error):
            continue
    return None


def _strip_modality(case_id: str) -> str:
    return re.sub(r"-(t1n|t1c|t2w|t2f|dwi)$", "", case_id, flags=re.IGNORECASE)


def load_report_splits(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for case_id in raw.get(split, []):
            subject_id = _strip_modality(case_id)
            previous = mapping.setdefault(subject_id, split)
            if previous != split:
                raise ValueError(f"Report subject {subject_id} occurs in {previous} and {split}")
    return mapping


def load_report_ids(report_meta: str | Path) -> dict[str, set[str]]:
    root = Path(report_meta)
    result: dict[str, set[str]] = {}
    for cohort, directory in REPORT_DIRECTORIES.items():
        path = root / directory / "global_finding.json"
        with path.open(encoding="utf-8") as handle:
            result[cohort] = set(json.load(handle))
    return result


def _header_metadata(paths: dict[str, str], segmentation: str | None) -> dict:
    references = list(paths.values()) + ([segmentation] if segmentation else [])
    images = [nib.load(path) for path in references]
    reference = images[0]
    orientations = ["".join(nib.aff2axcodes(image.affine)) for image in images]
    shapes = [tuple(int(value) for value in image.shape[:3]) for image in images]
    spacings = [tuple(float(value) for value in image.header.get_zooms()[:3]) for image in images]
    affine_match = all(np.allclose(image.affine, reference.affine, atol=1e-4) for image in images)
    return {
        "shape": list(shapes[0]),
        "spacing": list(spacings[0]),
        "orientation": orientations[0],
        "all_shapes_match": len(set(shapes)) == 1,
        "all_spacings_match": all(np.allclose(value, spacings[0], atol=1e-4) for value in spacings),
        "all_affines_match": affine_match,
    }


def _normalized_image_fingerprint(path: str) -> str:
    image = nib.as_closest_canonical(nib.load(path))
    array = np.asarray(image.dataobj, dtype=np.float32)
    steps = tuple(max(1, size // 32) for size in array.shape[:3])
    sample = array[:: steps[0], :: steps[1], :: steps[2]][:32, :32, :32]
    nonzero = sample[sample != 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, (1, 99))
        scale = max(float(high - low), 1e-6)
        sample = np.clip((sample - low) / scale, 0, 1)
    quantized = np.rint(sample * 255).astype(np.uint8)
    hasher = hashlib.sha256()
    hasher.update(np.asarray(array.shape[:3], dtype=np.int32).tobytes())
    hasher.update(quantized.tobytes())
    return hasher.hexdigest()


def _label_values(path: str) -> list[int]:
    return [int(value) for value in np.unique(np.asarray(nib.load(path).dataobj))]


def _voxel_equal(path_a: str, path_b: str) -> bool:
    left = np.asarray(nib.load(path_a).dataobj)
    right = np.asarray(nib.load(path_b).dataobj)
    return left.shape == right.shape and np.array_equal(left, right)


def resolve_content_duplicates(records: list[dict]) -> list[dict]:
    """Resolve image-identical patient aliases before splitting.

    A T1c fingerprint is intentionally sensitive enough to group likely aliases even when another
    sequence or annotation differs. Conflicting annotations quarantine the entire group because
    there is no defensible automatic choice. Exact image+mask copies retain the lexicographically
    first ID as the canonical case.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        record["eligible_for_split"] = bool(record["labeled"] and record["complete_images"])
        if record.get("image_fingerprint"):
            groups[record["image_fingerprint"]].append(record)
    resolutions = []
    for fingerprint, group in groups.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda item: (item["cohort"], item["subject_id"]))
        reference = group[0]
        segmentation_equal = all(
            _voxel_equal(reference["segmentation"], item["segmentation"])
            for item in group[1:]
            if reference.get("segmentation") and item.get("segmentation")
        )
        modalities_equal = all(
            _voxel_equal(reference["sequences"][modality], item["sequences"][modality])
            for item in group[1:]
            for modality in MODALITIES
        )
        members = [f"{item['cohort']}:{item['subject_id']}" for item in group]
        if segmentation_equal:
            reference["duplicate_resolution"] = "canonical_duplicate"
            for item in group[1:]:
                item["eligible_for_split"] = False
                item["duplicate_resolution"] = "excluded_duplicate_alias"
            action = "retain_one_exact" if modalities_equal else "retain_one_same_annotation"
        else:
            for item in group:
                item["eligible_for_split"] = False
                item["duplicate_resolution"] = "excluded_conflicting_duplicate_annotation"
            action = "quarantine_annotation_conflict"
        resolutions.append(
            {
                "fingerprint": fingerprint,
                "members": members,
                "modalities_equal": modalities_equal,
                "segmentation_equal": segmentation_equal,
                "action": action,
            }
        )
    return resolutions


def _iter_cases(dataset_root: Path) -> Iterable[tuple[str, str, Path, bool, bool]]:
    for cohort in COHORTS:
        cohort_root = dataset_root / COHORT_DIRECTORIES[cohort]
        corrections: dict[str, Path] = {}
        correction_branch = CORRECTION_BRANCH.get(cohort)
        if correction_branch:
            path = cohort_root / correction_branch
            if path.is_dir():
                corrections = {folder.name: folder for folder in path.iterdir() if folder.is_dir()}
        for branch in LABELED_BRANCHES[cohort]:
            branch_path = cohort_root / branch
            for folder in sorted(branch_path.iterdir()):
                if not folder.is_dir():
                    continue
                corrected = folder.name in corrections
                yield cohort, branch, corrections.get(folder.name, folder), True, corrected
        for branch in UNLABELED_BRANCHES[cohort]:
            branch_path = cohort_root / branch
            for folder in sorted(branch_path.iterdir()):
                if folder.is_dir():
                    yield cohort, branch, folder, False, False


def build_inventory(
    dataset_root: str | Path,
    report_meta: str | Path,
    report_split: str | Path,
    *,
    fingerprints: bool = False,
    deep_label_audit: bool = False,
) -> tuple[list[dict], dict]:
    root = Path(dataset_root)
    report_ids = load_report_ids(report_meta)
    report_splits = load_report_splits(report_split)
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for cohort, source_branch, folder, expected_label, corrected in _iter_cases(root):
        subject_id = folder.name
        key = (cohort, subject_id)
        if key in seen:
            raise ValueError(f"Duplicate canonical case emitted: {key}")
        seen.add(key)
        sequence_paths = {name: _resolve_file(folder, subject_id, name) for name in MODALITIES}
        segmentation = _resolve_file(folder, subject_id, "seg")
        complete_images = all(sequence_paths.values())
        labeled = segmentation is not None
        path_strings = {name: str(path) if path else None for name, path in sequence_paths.items()}
        metadata = {}
        if complete_images:
            metadata = _header_metadata(
                {name: value for name, value in path_strings.items() if value},
                str(segmentation) if segmentation else None,
            )
        record = {
            "subject_id": subject_id,
            "cohort": cohort,
            "source_branch": source_branch,
            "canonical_source_branch": folder.parent.name,
            "corrected_copy": corrected,
            "expected_labeled_branch": expected_label,
            "labeled": labeled,
            "complete_images": complete_images,
            "sequences": path_strings,
            "segmentation": str(segmentation) if segmentation else None,
            "has_report": subject_id in report_ids[cohort],
            "report_split": report_splits.get(subject_id),
            "identity_fingerprint": hashlib.sha256(f"{cohort}:{subject_id}".encode()).hexdigest(),
            **metadata,
        }
        if fingerprints and complete_images:
            record["image_fingerprint"] = _normalized_image_fingerprint(path_strings["t1c"])
        if deep_label_audit and labeled:
            record["label_values"] = _label_values(str(segmentation))
        records.append(record)

    duplicate_resolutions = resolve_content_duplicates(records) if fingerprints else []
    for record in records:
        record.setdefault("eligible_for_split", bool(record["labeled"] and record["complete_images"]))
    report_subjects = {record["subject_id"] for record in records if record["has_report"]}
    relevant_report_splits = {
        subject_id
        for subject_id in report_splits
        if subject_id.startswith(("BraTS-GLI-", "BraTS-MEN-", "BraTS-MET-"))
    }
    missing_report_cases = sorted(relevant_report_splits - report_subjects)
    labeled_records = [record for record in records if record["labeled"] and record["complete_images"]]
    unlabeled_records = [record for record in records if not record["labeled"] and record["complete_images"]]
    image_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.get("image_fingerprint"):
            image_groups[record["image_fingerprint"]].append(
                f"{record['cohort']}:{record['subject_id']}"
            )
    duplicate_groups = [values for values in image_groups.values() if len(values) > 1]
    audit = {
        "total_records": len(records),
        "labeled_complete": len(labeled_records),
        "unlabeled_complete": len(unlabeled_records),
        "labeled_by_cohort": dict(Counter(record["cohort"] for record in labeled_records)),
        "unlabeled_by_cohort": dict(Counter(record["cohort"] for record in unlabeled_records)),
        "orientation_labeled": dict(Counter(record.get("orientation") for record in labeled_records)),
        "corrected_copies": sum(record["corrected_copy"] for record in records),
        "report_subjects": sum(record["has_report"] for record in records),
        "report_splits": dict(Counter(record["report_split"] for record in records if record["has_report"])),
        "missing_report_cases": missing_report_cases,
        "incomplete_images": [
            f"{record['cohort']}:{record['subject_id']}"
            for record in records
            if not record["complete_images"]
        ],
        "unexpected_label_branch_mismatch": [
            f"{record['cohort']}:{record['subject_id']}"
            for record in records
            if record["expected_labeled_branch"] != record["labeled"]
        ],
        "alignment_failures": [
            f"{record['cohort']}:{record['subject_id']}"
            for record in records
            if record["complete_images"]
            and not (
                record.get("all_shapes_match")
                and record.get("all_spacings_match")
                and record.get("all_affines_match")
            )
        ],
        "content_duplicate_groups": duplicate_groups,
        "duplicate_resolutions": duplicate_resolutions,
        "eligible_labeled_after_duplicate_resolution": sum(
            record["eligible_for_split"] for record in records
        ),
        "duplicate_resolution_counts": dict(
            Counter(resolution["action"] for resolution in duplicate_resolutions)
        ),
        "label_value_sets": sorted(
            {tuple(record.get("label_values", [])) for record in labeled_records}
        ),
    }
    return records, audit
