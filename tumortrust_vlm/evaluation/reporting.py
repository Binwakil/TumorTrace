from __future__ import annotations

import re
from collections import Counter
from importlib.metadata import version

import numpy as np

WORD_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")
EVIDENCE_EXTRACTOR_VERSION = "structured_evidence_fields_v2"


def normalize_template(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def report_diversity(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "unique_report_fraction": 0.0,
            "unique_template_fraction": 0.0,
            "dominant_template_fraction": 0.0,
        }
    templates = [normalize_template(text) for text in texts]
    counts = Counter(templates)
    return {
        "unique_report_fraction": len(set(texts)) / len(texts),
        "unique_template_fraction": len(counts) / len(texts),
        "dominant_template_fraction": max(counts.values()) / len(texts),
    }


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1 if token == other else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float:
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    if not candidate_tokens or not reference_tokens:
        return 0.0
    common = _lcs_length(candidate_tokens, reference_tokens)
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu(candidate: str, reference: str, maximum_order: int = 4) -> float:
    """Sentence BLEU with clipped counts, add-one smoothing, and effective order.

    Effective order prevents a perfect short sentence from being assigned zero merely because it
    contains no four-grams. The exact policy is fixed here rather than delegated to a mutable
    external default.
    """
    if maximum_order < 1:
        raise ValueError("maximum_order must be positive")
    candidate_tokens = WORD_PATTERN.findall(candidate.lower())
    reference_tokens = WORD_PATTERN.findall(reference.lower())
    if not candidate_tokens or not reference_tokens:
        return 0.0
    order = min(maximum_order, len(candidate_tokens), len(reference_tokens))
    log_precisions = []
    for size in range(1, order + 1):
        candidate_counts = Counter(
            tuple(candidate_tokens[index : index + size])
            for index in range(len(candidate_tokens) - size + 1)
        )
        reference_counts = Counter(
            tuple(reference_tokens[index : index + size])
            for index in range(len(reference_tokens) - size + 1)
        )
        matches = sum(
            min(count, reference_counts[ngram]) for ngram, count in candidate_counts.items()
        )
        possible = sum(candidate_counts.values())
        log_precisions.append(np.log((matches + 1) / (possible + 1)))
    brevity_penalty = (
        1.0
        if len(candidate_tokens) >= len(reference_tokens)
        else np.exp(1 - len(reference_tokens) / len(candidate_tokens))
    )
    return float(brevity_penalty * np.exp(np.mean(log_precisions)))


def bertscore(
    candidates: list[str],
    references: list[str],
    *,
    model_type: str,
    device: str | None = None,
) -> dict:
    """Run an explicitly requested BERTScore model and return serializable case scores."""
    if len(candidates) != len(references):
        raise ValueError("BERTScore candidates and references must be paired")
    if not candidates:
        raise ValueError("BERTScore inputs must be non-empty")
    from bert_score import score

    precision, recall, f1 = score(
        candidates,
        references,
        model_type=model_type,
        device=device,
        verbose=False,
        rescale_with_baseline=False,
    )
    return {
        "model_type": model_type,
        "package_version": version("bert-score"),
        "rescale_with_baseline": False,
        "precision": precision.cpu().tolist(),
        "recall": recall.cpu().tolist(),
        "f1": f1.cpu().tolist(),
    }


def extract_evidence_fields(text: str) -> dict:
    lower = text.lower()
    result = {}
    volume_patterns = {
        "WT": (
            r"(?:whole[- ]tumor volume\s*(?:is|:)|\bwt\b(?:\s+volume)?\s*(?:is|:)?)"
            r"\s*([0-9.]+)\s*ml"
        ),
        "TC": (
            r"(?:tumor[- ]core volume\s*(?:is|:)|\btc\b(?:\s+volume)?\s*(?:is|:)?)"
            r"\s*([0-9.]+)\s*ml"
        ),
        "ET": (
            r"(?:enhancing[- ]tumor volume\s*(?:is|:)|\bet\b(?:\s+volume)?\s*(?:is|:)?)"
            r"\s*([0-9.]+)\s*ml"
        ),
        "SNFH": (
            r"(?:surrounding flair[- ]hyperintense volume\s*(?:is|:)|"
            r"\bsnfh\b(?:\s+volume)?\s*(?:is|:)?)\s*([0-9.]+)\s*ml"
        ),
    }
    for region, pattern in volume_patterns.items():
        match = re.search(pattern, lower)
        if match:
            result[f"volume_{region}"] = float(match.group(1))
    for laterality in ("bilateral", "midline", "left", "right", "none"):
        if re.search(rf"\b{laterality}\b", lower):
            result["laterality"] = laterality
            break
    for family in ("gli", "men", "met"):
        long_form = re.search(
            rf"tumor family(?: estimate)?\s*(?:(?:is|of)\s+|:\s*)?{family}\b", lower
        )
        family_first = re.search(rf"model-predicted\s+{family}\s+tumor family\b", lower)
        if long_form or family_first:
            result["family"] = family.upper()
            break
    component_match = re.search(r"\bcomponent count\s*(?:(?:is|:)\s*)?(\d+)\b", lower)
    if component_match is None:
        component_match = re.search(
            r"\b(\d+)\s+(?:[a-z-]+\s+){0,2}component(?:\(s\)|s)?\b", lower
        )
    if component_match:
        result["component_count"] = int(component_match.group(1))
    else:
        word_match = re.search(
            r"\b(one|single|two|three|four|five)(?:-|\s+(?:[a-z-]+\s+){0,2})components?\b",
            lower,
        )
        if word_match:
            result["component_count"] = {
                "one": 1,
                "single": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
            }[word_match.group(1)]
    result["referral"] = "review is recommended" in lower
    return result


def evidence_consistency(
    text: str, evidence: dict, volume_tolerance_ml: float = 0.11
) -> dict[str, float]:
    fields = extract_evidence_fields(text)
    unavailable = set(evidence.get("unavailable_fields", ()))
    supported = 0
    contradicted = 0
    expected = 0
    volumes = evidence.get("volumes_ml")
    if volumes is None:
        volumes = {
            region: payload["value_ml"] for region, payload in evidence.get("volumes", {}).items()
        }
    for region, value in volumes.items():
        if "volume" in unavailable or f"volume_{region}" in unavailable:
            continue
        expected += 1
        key = f"volume_{region}"
        if key in fields:
            if abs(fields[key] - value) <= volume_tolerance_ml:
                supported += 1
            else:
                contradicted += 1
    expected_fields = {
        "laterality": evidence.get("laterality"),
        "family": evidence.get("family", evidence.get("predicted_family")),
        "component_count": evidence.get("component_count"),
        "referral": evidence.get("referral"),
    }
    for key, value in expected_fields.items():
        availability_key = "tumor_family" if key == "family" else key
        if value is not None and availability_key not in unavailable:
            expected += 1
            if fields.get(key) == value:
                supported += 1
            elif key in fields:
                contradicted += 1
    return {
        "structured_field_recall": supported / expected if expected else 1.0,
        "structured_field_contradiction_rate": contradicted / max(1, len(fields)),
        "structured_field_unsupported_rate": contradicted / max(1, supported + contradicted),
    }


def intervention_accuracy(before: list[dict], after: list[dict], changed_field: str) -> float:
    if len(before) != len(after):
        raise ValueError("Intervention outputs must be paired")
    correct = []
    for left, right in zip(before, after):
        changed = left.get(changed_field) != right.get(changed_field)
        stable = all(
            left.get(key) == right.get(key)
            for key in set(left) | set(right)
            if key != changed_field
        )
        correct.append(changed and stable)
    return float(np.mean(correct)) if correct else float("nan")


def targeted_intervention_accuracy(
    before: list[dict],
    after: list[dict],
    expected: list[dict],
    changed_field: str,
    numeric_tolerance: float = 0.11,
) -> float:
    """Score a controlled intervention against the donor value and collateral stability."""
    if not (len(before) == len(after) == len(expected)):
        raise ValueError("Targeted intervention outputs must be paired")
    correct = []
    for left, right, target in zip(before, after, expected, strict=True):
        if changed_field not in target or left.get(changed_field) == target[changed_field]:
            continue
        observed = right.get(changed_field)
        desired = target[changed_field]
        if isinstance(desired, (int, float)) and isinstance(observed, (int, float)):
            target_match = abs(float(observed) - float(desired)) <= numeric_tolerance
        else:
            target_match = observed == desired
        stable = all(
            left.get(key) == right.get(key)
            for key in set(left) | set(right)
            if key != changed_field
        )
        correct.append(target_match and stable)
    return float(np.mean(correct)) if correct else float("nan")
