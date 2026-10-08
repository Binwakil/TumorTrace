from __future__ import annotations

import re
import unicodedata

from tumortrust_vlm.reporting.findings import (
    FINDING_SCHEMA_VERSION,
    extract_normalized_findings,
)

TARGET_SCHEMA_VERSION = "global_finding_nfkc_whitespace_v1"


def normalize_report_target(text: str) -> str:
    """Normalize encoding and whitespace without changing clinical wording."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("Report target is empty after normalization")
    return normalized


def assemble_report_records(
    feature_map: dict[str, dict],
    reports: dict[str, str],
    report_splits: dict[str, str],
    eligible_subject_ids: set[str],
    split: str,
) -> tuple[list[dict], list[str]]:
    """Join deployment-matched features to eligible reports under the frozen split."""
    requested = {subject_id for subject_id in reports if report_splits.get(subject_id) == split}
    eligible = requested & eligible_subject_ids
    excluded = sorted(requested - eligible_subject_ids)
    missing = sorted(eligible - feature_map.keys())
    if missing:
        raise ValueError(
            f"Missing deployment-matched features for {len(missing)} eligible report subjects"
        )
    records = []
    for subject_id in sorted(eligible):
        normalized_text = normalize_report_target(reports[subject_id])
        records.append(
            {
                **feature_map[subject_id],
                "report_text": normalized_text,
                "normalized_findings": extract_normalized_findings(normalized_text),
                "report_split": split,
                "report_target_schema": TARGET_SCHEMA_VERSION,
                "finding_target_schema": FINDING_SCHEMA_VERSION,
            }
        )
    return records, excluded
