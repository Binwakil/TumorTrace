from __future__ import annotations

import pytest

from tumortrust_vlm.reporting.llm_api import (
    assert_no_prohibited_content,
    evidence_prompt,
    extract_response_text,
    sanitize_evidence,
    swap_evidence_field,
)


def evidence() -> dict:
    return {
        "schema_version": "1.1",
        "subject_id": "private-subject",
        "tumor_family_probabilities": {"GLI": 0.8, "MEN": 0.1, "MET": 0.1},
        "predicted_family": "GLI",
        "volumes": {
            region: {"value_ml": float(index), "interval_90_ml": [0.0, float(index + 1)]}
            for index, region in enumerate(("WT", "TC", "ET", "SNFH"), start=1)
        },
        "component_count": 2,
        "laterality": "left",
        "segmentation_uncertainty": 0.2,
        "classification_uncertainty": 0.1,
        "referral": False,
        "referral_reasons": [],
        "unavailable_fields": [],
    }


def test_external_evidence_payload_is_allow_listed_and_identifier_free() -> None:
    payload = sanitize_evidence(evidence())
    assert "subject_id" not in payload
    assert "private-subject" not in evidence_prompt(evidence())


def test_prohibited_nested_identifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="Prohibited external payload key"):
        assert_no_prohibited_content({"safe": {"patient_id": "x"}})


def test_evidence_intervention_changes_only_requested_group() -> None:
    original = evidence()
    donor = evidence()
    donor["volumes"]["WT"]["value_ml"] = 99.0
    changed = swap_evidence_field(original, donor, "volume_WT")
    assert changed["volumes"]["WT"]["value_ml"] == 99.0
    assert changed["volumes"]["TC"] == original["volumes"]["TC"]


def test_raw_response_text_extraction() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Grounded findings."}],
            }
        ]
    }
    assert extract_response_text(response) == "Grounded findings."
