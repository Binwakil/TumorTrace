from __future__ import annotations

import re
from typing import Any

import numpy as np

FINDING_SCHEMA_VERSION = "brain_mri_normalized_findings_v1"
FINDING_REVIEW_STATUS = "rule_extracted_pending_radiologist_validation"

DIMENSION_PATTERN = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[*x×]\s*(\d+(?:\.\d+)?)"
    r"\s*[*x×]\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)\b",
    re.IGNORECASE,
)

ANATOMIC_PATTERNS = {
    "basal_ganglia": r"\bbasal gangli(?:a|on)\b",
    "brainstem": r"\b(?:brain\s*stem|midbrain|pons|medulla)\b",
    "cerebellum": r"\bcerebell(?:um|ar)\b",
    "corpus_callosum": r"\bcorpus callosum\b",
    "frontal_lobe": r"\bfrontal\b",
    "insular_region": r"\binsul(?:a|ar)\b",
    "midline_fissure": r"\b(?:longitudinal|interhemispheric) fissure\b",
    "occipital_lobe": r"\boccipital\b",
    "parietal_lobe": r"\bparietal\b",
    "sellar_region": r"\b(?:sellar|sella turcica|parasellar)\b",
    "skull_base": r"\b(?:cranial|skull) (?:base|plate)\b",
    "temporal_lobe": r"\btemporal\b",
    "thalamus": r"\bthalam(?:us|ic)\b",
    "ventricular": r"\bventric(?:le|les|ular)\b",
}


def _contains(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _dimensions(text: str) -> list[list[float]]:
    values = []
    for match in DIMENSION_PATTERN.finditer(text):
        triplet = [float(value) for value in match.groups()]
        if all(0 < value <= 300 for value in triplet):
            values.append(triplet)
    return values


def extract_normalized_findings(text: str) -> dict[str, Any]:
    """Conservatively map a brain-MRI narrative to reviewable finding fields.

    This deterministic parser creates weak structured targets; it does not replace the original
    report and its output is explicitly pending clinical validation.
    """
    lower = text.lower()
    dimensions = _dimensions(text)
    location_text = re.sub(r"\b(?:left|right)ward\b", "", lower)
    location_text = re.sub(
        r"\b(?:midline structures? (?:are |is )?)?(?:shifted|displaced|shift) to the "
        r"(?:left|right)\b",
        "",
        location_text,
    )
    lateralities = []
    if _contains(r"\b(?:bilateral|both)\b", location_text):
        lateralities.append("bilateral")
    for side in ("left", "right"):
        if _contains(rf"\b{side}\b", location_text):
            lateralities.append(side)
    if _contains(
        r"\b(?:midline (?:lesion|mass|focus|location)|across the midline|"
        r"longitudinal fissure|interhemispheric fissure)\b",
        location_text,
    ):
        lateralities.append("midline")

    if _contains(r"\b(?:multiple|multifocal|numerous|several|scattered)\b", lower):
        multiplicity = "multiple"
    elif _contains(r"\b(?:a|an|single|solitary)\s+(?:[^.]{0,30}\s)?(?:lesion|mass|focus|shadow)\b", lower):
        multiplicity = "single"
    else:
        multiplicity = None

    edema_absent = _contains(
        r"\bno\s+(?:surrounding |peritumoral |adjacent )?(?:brain (?:tissue )?)?edema\b",
        lower,
    )
    if edema_absent:
        edema = "absent"
    elif _contains(r"\b(?:extensive|severe|marked)\b[^.]{0,35}\bedema\b", lower):
        edema = "extensive"
    elif _contains(r"\b(?:mild|slight|minimal)\b[^.]{0,35}\bedema\b", lower):
        edema = "mild"
    elif _contains(r"\bedema(?:tous)?\b", lower):
        edema = "present_unspecified"
    else:
        edema = None

    shift_absent = _contains(
        r"\bno (?:displacement|shift) of (?:the )?midline|\bno midline shift|"
        r"\bno edema (?:or|and) midline shift|"
        r"\bmidline (?:structures )?(?:are )?(?:not shifted|without (?:a )?shift)",
        lower,
    )
    shift_present = _contains(
        r"\b(?:midline (?:structures )?(?:are |is )?(?:shifted|displaced)|"
        r"midline shift (?:is |was )?present|shift of (?:the )?midline|"
        r"(?:left|right)ward shift of midline)",
        lower,
    )
    if shift_absent:
        midline_shift = "absent"
    elif shift_present:
        if _contains(r"\b(?:leftward|to the left)\b", lower):
            midline_shift = "present_leftward"
        elif _contains(r"\b(?:rightward|to the right)\b", lower):
            midline_shift = "present_rightward"
        else:
            midline_shift = "present_unspecified"
    else:
        midline_shift = None

    if _contains(r"\bno (?:significant )?mass effect\b", lower):
        mass_effect = "absent"
    elif _contains(
        r"\b(?:mass effect|compress(?:ed|ion|ing)|narrow(?:ed|ing)|effacement)\b", lower
    ):
        mass_effect = "present"
    else:
        mass_effect = None

    enhancement_absent = _contains(r"\bno (?:obvious |significant )?enhancement\b", lower)
    enhancement_present = _contains(r"\benhanc(?:e|ed|ement|ing)\b", lower)
    enhancement_presence = (
        "absent" if enhancement_absent else "present" if enhancement_present else None
    )
    enhancement_patterns = []
    pattern_rules = {
        "ring": r"\bring(?:-like)? enhanc",
        "peripheral": r"\b(?:peripheral|rim) enhanc",
        "heterogeneous": r"\b(?:heterogeneous|uneven|nonuniform) enhanc",
        "homogeneous": r"\b(?:homogeneous|uniform) enhanc",
    }
    for label, pattern in pattern_rules.items():
        if _contains(pattern, lower):
            enhancement_patterns.append(label)
    if enhancement_present and _contains(
        r"\b(?:marked|obvious|significant|evident|strong)\b[^.]{0,30}\benhanc", lower
    ):
        enhancement_degree = "marked"
    elif enhancement_present and _contains(r"\bmoderate\b[^.]{0,30}\benhanc", lower):
        enhancement_degree = "moderate"
    elif enhancement_present and _contains(r"\b(?:mild|slight)\b[^.]{0,30}\benhanc", lower):
        enhancement_degree = "mild"
    else:
        enhancement_degree = None

    if _contains(r"\b(?:ill-defined|poorly defined|indistinct|unclear)\b", lower):
        margin = "ill_defined"
    elif _contains(r"\b(?:well-defined|well circumscribed|clear boundaries)\b", lower):
        margin = "well_defined"
    else:
        margin = None

    return {
        "schema_version": FINDING_SCHEMA_VERSION,
        "review_status": FINDING_REVIEW_STATUS,
        "multiplicity": multiplicity,
        "lateralities": lateralities,
        "anatomic_sites": [
            label for label, pattern in ANATOMIC_PATTERNS.items() if _contains(pattern, lower)
        ],
        "enhancement_presence": enhancement_presence,
        "enhancement_patterns": enhancement_patterns,
        "enhancement_degree": enhancement_degree,
        "edema": edema,
        "midline_shift": midline_shift,
        "mass_effect": mass_effect,
        "margin": margin,
        "reported_dimensions_mm": dimensions,
        "largest_reported_dimensions_mm": (
            max(dimensions, key=lambda values: float(np.prod(values))) if dimensions else None
        ),
    }


def finding_atoms(fields: dict[str, Any]) -> set[str]:
    """Flatten categorical finding fields for micro-averaged set scoring."""
    ignored = {
        "schema_version",
        "review_status",
        "reported_dimensions_mm",
        "largest_reported_dimensions_mm",
    }
    atoms = set()
    for field, value in fields.items():
        if field in ignored or value is None:
            continue
        if isinstance(value, list):
            atoms.update(f"{field}={item}" for item in value)
        else:
            atoms.add(f"{field}={value}")
    return atoms


def structured_laterality_metrics(
    candidates: list[str], reference_fields: list[dict[str, Any]]
) -> dict[str, float | int]:
    """Score the weakly extracted multi-label laterality field separately."""
    if len(candidates) != len(reference_fields):
        raise ValueError("Structured laterality inputs must be paired")
    true_positive = false_positive = false_negative = exact = 0
    for candidate, reference in zip(candidates, reference_fields, strict=True):
        predicted = set(extract_normalized_findings(candidate).get("lateralities", []))
        expected = set(reference.get("lateralities", []))
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        exact += predicted == expected
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "structured_laterality_true_positive": true_positive,
        "structured_laterality_false_positive": false_positive,
        "structured_laterality_false_negative": false_negative,
        "structured_laterality_micro_precision": precision,
        "structured_laterality_micro_recall": recall,
        "structured_laterality_micro_f1": f1,
        "structured_laterality_exact_match": exact / len(candidates) if candidates else 0.0,
    }


def structured_finding_metrics(
    candidates: list[str], reference_fields: list[dict[str, Any]]
) -> dict[str, float | int | str | None]:
    if len(candidates) != len(reference_fields):
        raise ValueError("Structured finding inputs must be paired")
    true_positive = false_positive = false_negative = 0
    dimension_errors = []
    reference_dimensions = candidate_dimensions = paired_dimensions = 0
    for candidate, reference in zip(candidates, reference_fields, strict=True):
        prediction = extract_normalized_findings(candidate)
        predicted_atoms = finding_atoms(prediction)
        reference_atoms = finding_atoms(reference)
        true_positive += len(predicted_atoms & reference_atoms)
        false_positive += len(predicted_atoms - reference_atoms)
        false_negative += len(reference_atoms - predicted_atoms)
        expected_size = reference.get("largest_reported_dimensions_mm")
        predicted_size = prediction.get("largest_reported_dimensions_mm")
        reference_dimensions += expected_size is not None
        candidate_dimensions += predicted_size is not None
        if expected_size is not None and predicted_size is not None:
            paired_dimensions += 1
            expected_array = np.sort(np.asarray(expected_size, dtype=float))
            predicted_array = np.sort(np.asarray(predicted_size, dtype=float))
            dimension_errors.extend(np.abs(expected_array - predicted_array).tolist())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "finding_schema_version": FINDING_SCHEMA_VERSION,
        "finding_schema_review_status": FINDING_REVIEW_STATUS,
        "structured_finding_true_positive": true_positive,
        "structured_finding_false_positive": false_positive,
        "structured_finding_false_negative": false_negative,
        "structured_finding_micro_precision": precision,
        "structured_finding_micro_recall": recall,
        "structured_finding_micro_f1": f1,
        "dimension_reference_cases": reference_dimensions,
        "dimension_candidate_cases": candidate_dimensions,
        "dimension_paired_cases": paired_dimensions,
        "dimension_extraction_recall": (
            paired_dimensions / reference_dimensions if reference_dimensions else None
        ),
        "dimension_mae_mm": float(np.mean(dimension_errors)) if dimension_errors else None,
    }
