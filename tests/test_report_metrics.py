import pytest

from scripts.evaluate_clinical_text_metrics import summarize_radeval_results
from tumortrust_vlm.evaluation.reporting import (
    bleu,
    evidence_consistency,
    extract_evidence_fields,
    intervention_accuracy,
    report_diversity,
    rouge_l,
    targeted_intervention_accuracy,
)


def test_bleu_identity_and_mismatch() -> None:
    reference = "Enhancing tumor is present in the left frontal lobe."
    assert bleu(reference, reference, 1) == 1.0
    assert bleu(reference, reference, 4) == 1.0
    assert bleu("No acute finding.", reference, 4) < 0.25
    assert bleu("", reference, 4) == 0.0


def test_evidence_consistency_accepts_versioned_evidence_card() -> None:
    evidence = {
        "volumes": {
            "WT": {"value_ml": 12.3},
            "TC": {"value_ml": 4.0},
            "ET": {"value_ml": 2.0},
            "SNFH": {"value_ml": 8.3},
        },
        "laterality": "left",
        "predicted_family": "GLI",
        "referral": False,
    }
    text = (
        "Model-predicted tumor family: GLI. Whole-tumor volume is 12.3 mL. "
        "Tumor-core volume is 4.0 mL. Enhancing-tumor volume is 2.0 mL. "
        "Surrounding FLAIR-hyperintense volume is 8.3 mL. Spatial distribution: left."
    )

    result = evidence_consistency(text, evidence)

    assert result["structured_field_recall"] == 1.0
    assert result["structured_field_contradiction_rate"] == 0.0


def test_report_factuality_and_diversity():
    text = (
        "Model-predicted tumor family: GLI (80.0%). Whole-tumor volume is 10.0 mL. "
        "Spatial distribution: left. Specialist review is recommended."
    )
    fields = extract_evidence_fields(text)
    assert fields["volume_WT"] == 10.0
    assert fields["laterality"] == "left"
    consistency = evidence_consistency(
        text, {"volumes_ml": {"WT": 10.0}, "laterality": "left", "family": "GLI", "referral": True}
    )
    assert consistency["structured_field_recall"] == 1.0
    assert report_diversity([text, text])["dominant_template_fraction"] == 1.0


def test_evidence_extractor_accepts_concise_llm_phrasing() -> None:
    text = (
        "Automated segmentation identifies 1 left-sided component, with region volumes of "
        "ET 32.052 mL, SNFH 148.335 mL, TC 126.473 mL, and WT 274.808 mL. "
        "The model-predicted tumor family is GLI."
    )
    fields = extract_evidence_fields(text)
    assert fields["volume_WT"] == 274.808
    assert fields["volume_TC"] == 126.473
    assert fields["volume_ET"] == 32.052
    assert fields["volume_SNFH"] == 148.335
    assert fields["family"] == "GLI"
    assert fields["laterality"] == "left"
    assert fields["component_count"] == 1


@pytest.mark.parametrize(
    ("text", "family", "laterality", "components"),
    [
        ("Model-predicted tumor family MEN. Single-component right lesion.", "MEN", "right", 1),
        ("Model-predicted tumor family of GLI. Component count is 2.", "GLI", None, 2),
        ("Model-predicted MET tumor family. Laterality is none; component count 0.", "MET", "none", 0),
    ],
)
def test_evidence_extractor_accepts_equivalent_family_and_component_grammar(
    text: str, family: str, laterality: str | None, components: int
) -> None:
    fields = extract_evidence_fields(text)
    assert fields["family"] == family
    assert fields.get("laterality") == laterality
    assert fields["component_count"] == components
    assert rouge_l(text, text) == 1.0


def test_targeted_intervention_requires_changed_field_and_stable_others():
    before = {"family": "GLI", "volume_WT": 10.0, "laterality": "left"}
    correct = {"family": "MEN", "volume_WT": 10.0, "laterality": "left"}
    collateral = {"family": "MEN", "volume_WT": 11.0, "laterality": "left"}

    assert intervention_accuracy([before], [correct], "family") == 1.0
    assert intervention_accuracy([before], [collateral], "family") == 0.0
    assert targeted_intervention_accuracy([before], [correct], [{"family": "MEN"}], "family") == 1.0
    assert (
        targeted_intervention_accuracy([before], [collateral], [{"family": "MEN"}], "family") == 0.0
    )


def test_clinical_metric_summary_preserves_radcliq_direction() -> None:
    result = summarize_radeval_results(
        {
            "radgraph_simple": [0.5, 1.0],
            "ratescore": [0.4, 0.8],
            "radcliq_v1": [0.25, 0.75],
        }
    )
    assert result["radgraph_simple"] == 0.75
    assert result["ratescore"] == pytest.approx(0.6)
    assert result["radcliq_v1_raw_mean"] == 0.5
    assert result["radcliq_v1_inverse_mean"] == 2.0
