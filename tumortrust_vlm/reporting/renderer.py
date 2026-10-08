from __future__ import annotations

from tumortrust_vlm.reporting.evidence import EvidenceCard

VOLUME_LABELS = {
    "WT": "Whole-tumor",
    "TC": "Tumor-core",
    "ET": "Enhancing-tumor",
    "SNFH": "Surrounding FLAIR-hyperintense",
}


def _volume_sentence(name: str, card: EvidenceCard) -> str:
    evidence = card.volumes[name]
    label = VOLUME_LABELS[name]
    sentence = f"{label} volume is {evidence.value_ml:.1f} mL"
    if evidence.interval_90_ml is not None:
        sentence += f" (90% interval {evidence.interval_90_ml[0]:.1f}–{evidence.interval_90_ml[1]:.1f} mL)"
    return sentence + "."


def _is_unavailable(card: EvidenceCard, field: str) -> bool:
    unavailable = set(card.unavailable_fields)
    return field in unavailable or field.split("_", maxsplit=1)[0] in unavailable


def render_findings(card: EvidenceCard) -> str:
    lines = []
    if _is_unavailable(card, "tumor_family"):
        lines.append("Tumor-family estimate is unavailable; no family claim is made.")
    else:
        probability = card.tumor_family_probabilities[card.predicted_family]
        lines.append(
            f"Model-predicted tumor family: {card.predicted_family} ({probability:.1%})."
        )
    available_volumes = [
        region
        for region in ("WT", "TC", "ET", "SNFH")
        if not _is_unavailable(card, f"volume_{region}")
    ]
    lines.extend(_volume_sentence(region, card) for region in available_volumes)
    if not available_volumes:
        lines.append("Quantitative tumor volumes are unavailable; no volume claim is made.")

    laterality_available = not _is_unavailable(card, "laterality")
    components_available = not _is_unavailable(card, "component_count")
    if laterality_available and components_available:
        lines.append(
            f"Spatial distribution: {card.laterality}; {card.component_count} component(s) "
            "above the prespecified size threshold."
        )
    elif laterality_available:
        lines.append(
            f"Spatial distribution: {card.laterality}; component count is unavailable."
        )
    elif components_available:
        lines.append(
            f"Spatial distribution is unavailable; {card.component_count} component(s) "
            "above the prespecified size threshold."
        )
    else:
        lines.append("Spatial distribution and component count are unavailable.")
    if card.referral:
        lines.append("Model uncertainty is elevated; specialist review is recommended before these findings are used.")
    else:
        lines.append("These automated research findings require specialist review and are not treatment advice.")
    return " ".join(lines)


def validate_rendered_values(card: EvidenceCard, text: str) -> None:
    for region, evidence in card.volumes.items():
        if _is_unavailable(card, f"volume_{region}"):
            continue
        token = f"{evidence.value_ml:.1f} mL"
        if token not in text:
            raise AssertionError(f"Rendered report omitted exact evidence value: {token}")
    for field in card.unavailable_fields:
        if field.startswith("volume_"):
            region = field.removeprefix("volume_")
            label = VOLUME_LABELS.get(region)
            if label is not None and f"{label} volume" in text:
                raise AssertionError(f"Renderer exposed unavailable evidence field: {field}")
    if "treatment recommendation" in text.lower():
        raise AssertionError("Renderer produced treatment advice")
