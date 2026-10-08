from __future__ import annotations

import copy
import json
from typing import Any

PROMPT_VERSION = "tumortrust_evidence_verbalizer_v1"
PROHIBITED_KEYS = {
    "data_path",
    "image",
    "normalized_findings",
    "patient_id",
    "report_text",
    "source_path",
    "subject_id",
    "visual_tokens",
}
TRANSMITTED_KEYS = {
    "classification_uncertainty",
    "component_count",
    "laterality",
    "predicted_family",
    "referral",
    "referral_reasons",
    "schema_version",
    "segmentation_uncertainty",
    "tumor_family_probabilities",
    "unavailable_fields",
    "volumes",
}

SYSTEM_INSTRUCTIONS = """You are a research brain-MRI evidence verbalizer. Generate one concise
findings paragraph grounded exclusively in the supplied structured evidence. Label tumor family as
model-predicted and preserve every available numeric value exactly as supplied. Mention the four
region volumes, 90% intervals when available, laterality, and component count. Do not infer anatomic
site, enhancement, edema, mass effect, dimensions, diagnosis, prognosis, or management when these
are absent from the evidence. Do not provide treatment advice. End with: These automated research
findings require specialist review and are not treatment advice."""


def sanitize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the allow-listed, identifier-free payload permitted for external inference."""
    payload = {key: copy.deepcopy(evidence[key]) for key in TRANSMITTED_KEYS if key in evidence}
    assert_no_prohibited_content(payload)
    return payload


def assert_no_prohibited_content(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.lower()
            if normalized in PROHIBITED_KEYS or normalized.endswith("_id"):
                raise ValueError(f"Prohibited external payload key: {path}.{key}")
            assert_no_prohibited_content(value, f"{path}.{key}")
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            assert_no_prohibited_content(value, f"{path}[{index}]")


def evidence_prompt(evidence: dict[str, Any]) -> str:
    payload = sanitize_evidence(evidence)
    return (
        f"Prompt version: {PROMPT_VERSION}\n"
        "Write the findings paragraph from this evidence JSON only:\n"
        f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
    )


def swap_evidence_field(
    evidence: dict[str, Any], donor: dict[str, Any], field: str
) -> dict[str, Any]:
    """Apply the same one-field donor intervention used by the matched 7B probes."""
    changed = copy.deepcopy(evidence)
    if field == "family":
        changed["predicted_family"] = donor["predicted_family"]
        changed["tumor_family_probabilities"] = copy.deepcopy(
            donor["tumor_family_probabilities"]
        )
    elif field.startswith("volume_"):
        region = field.removeprefix("volume_")
        changed["volumes"][region] = copy.deepcopy(donor["volumes"][region])
    elif field in {"laterality", "component_count", "referral"}:
        changed[field] = copy.deepcopy(donor[field])
        if field == "referral":
            changed["referral_reasons"] = copy.deepcopy(donor.get("referral_reasons", []))
    else:
        raise ValueError(f"Unsupported evidence intervention: {field}")
    return changed


def extract_response_text(response: dict[str, Any]) -> str:
    """Extract assistant text from a raw Responses API payload without relying on an SDK."""
    parts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("Responses API payload contains no output text")
    return text
