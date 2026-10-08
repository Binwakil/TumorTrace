#!/usr/bin/env python
"""Build a deidentified development-only terminology review packet."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.reporting.evidence import EvidenceCard
from tumortrust_vlm.reporting.renderer import render_findings
from tumortrust_vlm.utils import atomic_json_dump, sha256_file

INPUT = ROOT / "artifacts/private/trust/m1_val_features_mc10.json"
PRIVATE_OUTPUT = ROOT / "artifacts/private/radiologist_terminology_examples.json"
DOCUMENT = ROOT / "docs/RADIOLOGIST_TERMINOLOGY_PACKET.md"


def rank(subject_id: str) -> str:
    return hashlib.sha256(f"terminology_review_v1:{subject_id}".encode()).hexdigest()


def main() -> None:
    records = json.loads(INPUT.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Terminology example source is empty")
    if any(
        not record.get("subject_provenance", {}).get(
            "excluded_from_checkpoint_training", False
        )
        for record in records
    ):
        raise ValueError("Terminology examples are not proven development-held-out")

    selected = []
    for cohort in ("GLI", "MEN", "MET"):
        cohort_records = sorted(
            (record for record in records if record["cohort"] == cohort),
            key=lambda record: rank(record["subject_id"]),
        )[:2]
        if len(cohort_records) != 2:
            raise ValueError(f"Need two terminology examples for {cohort}")
        for index, record in enumerate(cohort_records, start=1):
            alias = f"{cohort}-{index}"
            card = EvidenceCard.from_dict(record["evidence"])
            selected.append(
                {
                    "alias": alias,
                    "subject_id": record["subject_id"],
                    "cohort": cohort,
                    "selection_hash": rank(record["subject_id"]),
                    "evidence": record["evidence"],
                    "rendered_findings": render_findings(card),
                }
            )
    atomic_json_dump(
        {
            "scope": "development validation only",
            "selection_policy": "two lowest SHA-256 ranks per cohort using terminology_review_v1 salt; no metric-based selection",
            "source_sha256": sha256_file(INPUT),
            "records": selected,
            "locked_test_opened": False,
        },
        PRIVATE_OUTPUT,
    )

    lines = [
        "# TumorTrust-VLM radiologist terminology and wording packet",
        "",
        "## Purpose and scope",
        "",
        "This packet requests terminology approval only. It is not the blinded reader study and does not expose the locked final test. The six examples were selected without looking at performance: two development-validation subjects per cohort with the lowest salted SHA-256 ranks. Internal subject identifiers remain only in the private provenance JSON.",
        "",
        "For every item, please mark **approve**, **revise**, or **reject**, and supply preferred wording when revision is requested.",
        "",
        "## 1. Region terminology",
        "",
        "| Internal field | Proposed reader-facing term | Intended meaning | Decision |",
        "|---|---|---|---|",
        "| WT | Whole-tumor region | Union of all ontology-approved tumor-related labels; not synonymous with viable tumor mass | ☐ Approve ☐ Revise ☐ Reject |",
        "| TC | Tumor-core region | Enhancing and non-enhancing/necrotic core labels where defined by the source ontology | ☐ Approve ☐ Revise ☐ Reject |",
        "| ET | Enhancing-tumor region | Contrast-enhancing tumor label | ☐ Approve ☐ Revise ☐ Reject |",
        "| SNFH | Surrounding FLAIR-hyperintense region | Peritumoral FLAIR hyperintensity; avoid calling it edema when the image label cannot distinguish edema from infiltrative change | ☐ Approve ☐ Revise ☐ Reject |",
        "",
        "**Proposal:** retain volumes in mL and omit a brain-normalized burden fraction from clinical-facing text until its interpretation is approved.",
        "",
        "Decision: ☐ Approve volumes only ☐ Add burden fraction with revised wording ☐ Reject quantitative burden wording",
        "",
        "## 2. Spatial-distribution policy",
        "",
        "The current deterministic rule uses canonical left-right coordinates. A central band spans 5% of image width around the midline. A case is `midline` when at least 25% of tumor voxels fall in that band; `bilateral` when both sides are occupied and the smaller side has at least 20% of the larger side's voxel count; otherwise it is `left` or `right`. Empty masks return `none`.",
        "",
        "Decision: ☐ Approve thresholds and labels ☐ Revise thresholds ☐ Use `crosses midline` instead of `bilateral` ☐ Add `indeterminate` rule",
        "",
        "Requested revisions: ________________________________",
        "",
        "## 3. Referral wording",
        "",
        "> Model uncertainty is elevated; specialist review is recommended before these findings are used.",
        "",
        "Non-referred cases currently end with:",
        "",
        "> These automated research findings require specialist review and are not treatment advice.",
        "",
        "Decision: ☐ Approve both ☐ Revise referred wording ☐ Revise routine disclaimer ☐ Do not expose a binary referral statement",
        "",
        "## 4. Error-review definitions",
        "",
        "| Label | Proposed operational definition | Decision |",
        "|---|---|---|",
        "| Unsupported | A report assertion has no corresponding supplied evidence field or image-supported finding | ☐ Approve ☐ Revise |",
        "| Contradictory | A report assertion conflicts with the supplied evidence in family, laterality, presence, direction, or quantitative value | ☐ Approve ☐ Revise |",
        "| Incomplete | A clinically relevant supplied finding is omitted without an explicit abstention | ☐ Approve ☐ Revise |",
        "| Significant | An error or omission could alter diagnostic interpretation or recommended follow-up | ☐ Approve ☐ Revise |",
        "| Harmful | An error or omission could plausibly cause inappropriate treatment, delayed urgent care, or failure to obtain required specialist review | ☐ Approve ☐ Revise |",
        "",
        "Please define a harmful-underestimation threshold, if a numeric rule is clinically defensible: ________________________________",
        "",
        "## 5. Development-only wording examples",
        "",
        "These examples are included solely to make terminology concrete. They are not cherry-picked successes and are not performance evidence.",
        "",
    ]
    for record in selected:
        lines.extend(
            [
                f"### {record['alias']}",
                "",
                record["rendered_findings"],
                "",
                "Terminology issues or preferred rewrite: ________________________________",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. Reader-study logistics requiring collaborator decision",
            "",
            "- Number and experience level of readers: ____________________",
            "- Pilot and final sample sizes: ____________________",
            "- Washout period between report conditions: ____________________",
            "- Whether adjudication is required: ____________________",
            "- Maximum acceptable reading burden per session: ____________________",
            "",
            "The full blinded sample will not be drawn until reporter conditions, case-selection policy, and final thresholds are frozen.",
        ]
    )
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DOCUMENT)
    print(PRIVATE_OUTPUT)


if __name__ == "__main__":
    main()
