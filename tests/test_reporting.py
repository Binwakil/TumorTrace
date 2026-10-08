import json

import numpy as np
import pytest
import torch

from scripts.analyze_finding_review import parse_review_value, score_review
from scripts.train_reporter import main as train_reporter_main
from tumortrust_vlm.models.vicuna_reporter import local_decoder_family
from tumortrust_vlm.reporting.composition import compose_reporter_features
from tumortrust_vlm.reporting.dataset import validate_reporter_record_sets
from tumortrust_vlm.reporting.evidence import build_evidence_card
from tumortrust_vlm.reporting.findings import (
    FINDING_SCHEMA_VERSION,
    extract_normalized_findings,
    structured_finding_metrics,
    structured_laterality_metrics,
)
from tumortrust_vlm.reporting.renderer import render_findings, validate_rendered_values
from tumortrust_vlm.reporting.targets import (
    TARGET_SCHEMA_VERSION,
    assemble_report_records,
    normalize_report_target,
)
from tumortrust_vlm.reporting.vicuna_dataset import (
    VicunaReporterDataset,
    collate_vicuna,
)


def test_deterministic_renderer_exact_trace_and_no_advice():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[2:12, 2:12, 2:12] = 2
    mask[4:8, 4:8, 4:8] = 3
    card = build_evidence_card(
        "subject",
        mask,
        (1.0, 1.0, 1.0),
        {"GLI": 0.8, "MEN": 0.1, "MET": 0.1},
        segmentation_uncertainty=0.9,
        referral_threshold=0.5,
    )
    text = render_findings(card)
    validate_rendered_values(card, text)
    assert card.referral
    assert card.from_dict(card.to_dict()) == card
    assert "specialist review" in text
    assert "treatment advice" not in text.lower()


def test_renderer_abstains_from_unavailable_evidence():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[2:12, 2:12, 2:12] = 2
    card = build_evidence_card(
        "subject",
        mask,
        (1.0, 1.0, 1.0),
        {"GLI": 0.8, "MEN": 0.1, "MET": 0.1},
        unavailable_fields=("tumor_family", "volume_ET", "laterality"),
    )

    text = render_findings(card)
    validate_rendered_values(card, text)

    assert "family claim is made" in text
    assert "Model-predicted tumor family" not in text
    assert "Enhancing-tumor volume" not in text
    assert "Spatial distribution is unavailable" in text


def test_referral_threshold_retains_values_at_frozen_coverage_boundary():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    card = build_evidence_card(
        "subject",
        mask,
        (1.0, 1.0, 1.0),
        {"GLI": 0.8, "MEN": 0.1, "MET": 0.1},
        segmentation_uncertainty=0.5,
        segmentation_referral_threshold=0.5,
    )

    assert not card.referral


def test_report_target_normalization_is_semantics_preserving():
    assert normalize_report_target("  Finding:\n  left  lesion. ") == "Finding: left lesion."


def test_normalized_finding_schema_is_conservative_and_versioned():
    text = (
        "Multiple lesions involve the left frontal and right temporal lobes, with marked "
        "ring-like enhancement, extensive edema, and dimensions of 20*30*40 mm. "
        "There is a leftward shift of midline structures."
    )
    fields = extract_normalized_findings(text)
    assert fields["schema_version"] == FINDING_SCHEMA_VERSION
    assert fields["multiplicity"] == "multiple"
    assert fields["lateralities"] == ["left", "right"]
    assert fields["anatomic_sites"] == ["frontal_lobe", "temporal_lobe"]
    assert fields["enhancement_patterns"] == ["ring"]
    assert fields["enhancement_degree"] == "marked"
    assert fields["edema"] == "extensive"
    assert fields["midline_shift"] == "present_leftward"
    assert fields["largest_reported_dimensions_mm"] == [20.0, 30.0, 40.0]


def test_normalized_findings_handle_negation_and_score_micro_f1():
    reference = extract_normalized_findings(
        "A right frontal lesion measures 10 x 20 x 30 mm. No edema or midline shift is present."
    )
    assert reference["edema"] == "absent"
    assert reference["midline_shift"] == "absent"
    result = structured_finding_metrics(
        ["A right frontal lesion measures 10 x 20 x 30 mm. No edema or midline shift is present."],
        [reference],
    )
    assert result["structured_finding_micro_f1"] == 1.0
    assert result["dimension_mae_mm"] == 0.0


def test_finding_review_analysis_scores_scalars_lists_and_dimensions():
    automatic = extract_normalized_findings(
        "A left frontal lesion measures 10*20*30 mm with ring enhancement and no midline shift."
    )
    reviewer = {
        "case_code": "FFR-001",
        **{f"reviewer_{field}": "unspecified" for field in automatic if field not in {"schema_version", "review_status", "reported_dimensions_mm"}},
    }
    reviewer.update(
        {
            "reviewer_multiplicity": "single",
            "reviewer_lateralities": "left",
            "reviewer_anatomic_sites": "frontal_lobe",
            "reviewer_enhancement_presence": "present",
            "reviewer_enhancement_patterns": "ring",
            "reviewer_midline_shift": "absent",
            "reviewer_largest_reported_dimensions_mm": "10*20*30",
        }
    )
    result = score_review(
        [reviewer], {"FFR-001": {"automatic_findings": automatic}}
    )
    assert parse_review_value("lateralities", "left; right") == ["left", "right"]
    assert result["scalar_exact_accuracy"] == 1.0
    assert result["list_micro_f1"] == 1.0
    assert result["by_field"]["largest_reported_dimensions_mm"]["mae_mm"] == 0.0


def test_report_join_requires_only_frozen_eligible_subjects():
    features = {"eligible": {"subject_id": "eligible", "evidence_vector": [1.0]}}
    reports = {"eligible": "Left lesion.", "quarantined": "Right lesion."}
    splits = {"eligible": "train", "quarantined": "train"}

    records, excluded = assemble_report_records(features, reports, splits, {"eligible"}, "train")

    assert len(records) == 1
    assert records[0]["report_target_schema"] == TARGET_SCHEMA_VERSION
    assert records[0]["finding_target_schema"] == FINDING_SCHEMA_VERSION
    assert records[0]["normalized_findings"]["lateralities"] == ["left"]
    assert excluded == ["quarantined"]


def test_report_join_rejects_missing_eligible_features():
    with pytest.raises(ValueError, match="1 eligible report subjects"):
        assemble_report_records(
            {},
            {"eligible": "Left lesion."},
            {"eligible": "val"},
            {"eligible"},
            "val",
        )


def test_compose_reporter_features_uses_segmentation_geometry_and_classifier_probabilities():
    evidence = {
        "schema_version": "1.1",
        "subject_id": "case",
        "tumor_family_probabilities": {"GLI": 0.8, "MEN": 0.1, "MET": 0.1},
        "predicted_family": "GLI",
        "volumes": {
            region: {"value_ml": float(index), "interval_90_ml": None, "interval_95_ml": None}
            for index, region in enumerate(("WT", "TC", "ET", "SNFH"), start=1)
        },
        "component_count": 2,
        "laterality": "left",
        "segmentation_uncertainty": 0.2,
        "classification_uncertainty": 0.4,
        "referral": False,
        "referral_reasons": [],
        "unavailable_fields": [],
    }
    base = {
        "subject_id": "case",
        "cohort": "GLI",
        "source_branch": "source",
        "evidence": evidence,
        "evidence_vector": [0.0] * 16,
        "visual_tokens": [[1.0, 2.0]],
        "global_features": [3.0],
        "uncertainty": {"segmentation_predictive_entropy_p95": 0.2},
        "inference_provenance": {"checkpoint_sha256": ["seg"]},
        "subject_provenance": {
            "feature_extraction_split": "test",
            "excluded_from_checkpoint_training": True,
        },
    }
    classification = {
        **base,
        "evidence": {
            **evidence,
            "tumor_family_probabilities": {"GLI": 0.1, "MEN": 0.2, "MET": 0.7},
            "predicted_family": "MET",
            "classification_uncertainty": 0.9,
        },
        "visual_tokens": [[9.0, 9.0]],
        "uncertainty": {"classification_predictive_entropy": 0.9},
        "inference_provenance": {"checkpoint_sha256": ["cls"]},
    }

    result = compose_reporter_features([base], [classification])[0]

    assert result["visual_tokens"] == [[1.0, 2.0]]
    assert result["evidence"]["predicted_family"] == "MET"
    assert result["evidence_vector"][:3] == [0.1, 0.2, 0.7]
    assert result["evidence_vector"][14] == 0.9
    assert result["inference_provenance"]["segmentation_recipe"][
        "checkpoint_sha256"
    ] == ["seg"]


def test_compose_reporter_features_rejects_recipe_subject_mismatch():
    record = {
        "subject_id": "case",
        "subject_provenance": {"excluded_from_checkpoint_training": True},
    }
    with pytest.raises(ValueError, match="subject sets differ"):
        compose_reporter_features([record], [])


def test_reporter_records_require_proven_oof_evidence():
    record = {
        "subject_id": "train",
        "report_split": "train",
        "evidence_vector": [1.0],
        "visual_tokens": [[1.0]],
        "report_target_schema": TARGET_SCHEMA_VERSION,
        "finding_target_schema": FINDING_SCHEMA_VERSION,
        "subject_provenance": {"excluded_from_checkpoint_training": False},
    }
    validation = {
        **record,
        "subject_id": "val",
        "report_split": "val",
        "subject_provenance": {"excluded_from_checkpoint_training": True},
    }
    with pytest.raises(ValueError, match="not proven out-of-fold"):
        validate_reporter_record_sets([record], [validation])


def test_compact_reporter_resume_is_exact(tmp_path, monkeypatch):
    def records(split: str, count: int) -> list[dict]:
        return [
            {
                "subject_id": f"{split}-{index}",
                "report_split": split,
                "report_text": f"A left frontal lesion measures {index + 1} mm.",
                "evidence_vector": [float(index), 1.0, 0.5],
                "visual_tokens": [[float(index), 1.0], [0.5, 0.25]],
                "report_target_schema": TARGET_SCHEMA_VERSION,
                "finding_target_schema": FINDING_SCHEMA_VERSION,
                "subject_provenance": {"excluded_from_checkpoint_training": True},
            }
            for index in range(count)
        ]

    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    train_path.write_text(json.dumps(records("train", 6)))
    val_path.write_text(json.dumps(records("val", 3)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def train(output, epochs: int, resume=None) -> None:
        arguments = [
            "train_reporter.py",
            "--train-records",
            str(train_path),
            "--val-records",
            str(val_path),
            "--output",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            "2",
            "--hidden-dim",
            "8",
            "--embedding-dim",
            "8",
            "--maximum-length",
            "24",
        ]
        if resume:
            arguments.extend(("--resume", str(resume)))
        monkeypatch.setattr("sys.argv", arguments)
        train_reporter_main()

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train(uninterrupted, 4)
    train(resumed, 2)
    train(resumed, 4, resumed / "last.pt")
    full_checkpoint = torch.load(uninterrupted / "last.pt", weights_only=False)
    resumed_checkpoint = torch.load(resumed / "last.pt", weights_only=False)
    for name, tensor in full_checkpoint["model"].items():
        assert torch.equal(tensor, resumed_checkpoint["model"][name])


def test_vicuna_report_targets_mask_prompt_and_keep_eos(tmp_path):
    class Tokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens):
            if "Generate a concise" in text:
                return [1, 10, 11]
            return [20, 21, 22, 23]

    path = tmp_path / "records.json"
    path.write_text(
        '[{"subject_id":"case","report_text":"finding","evidence_vector":[1.0,2.0],'
        '"visual_tokens":[[1.0,2.0]]}]',
        encoding="utf-8",
    )
    dataset = VicunaReporterDataset(path, Tokenizer(), maximum_length=6)
    item = dataset[0]

    assert item["labels"][:3].tolist() == [-100, -100, -100]
    assert item["labels"][-1].item() == 2
    batch = collate_vicuna([item], pad_token_id=0)
    assert torch.equal(batch["attention_mask"], torch.ones_like(batch["attention_mask"]))


def test_local_decoder_family_supports_locked_llava_variants(tmp_path):
    llama = tmp_path / "llama"
    mistral = tmp_path / "mistral"
    unsupported = tmp_path / "unsupported"
    for path, model_type in (
        (llama, "llava"),
        (mistral, "llava_mistral"),
        (unsupported, "other"),
    ):
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8"
        )
    assert local_decoder_family(llama) == "llama"
    assert local_decoder_family(mistral) == "mistral"
    with pytest.raises(ValueError, match="Unsupported local LLaVA model_type"):
        local_decoder_family(unsupported)


def test_structured_laterality_metrics_scores_multi_label_fields():
    candidates = ["A lesion is present in the left frontal lobe.", "Right temporal mass."]
    references = [
        {"lateralities": ["left"]},
        {"lateralities": ["bilateral"]},
    ]
    metrics = structured_laterality_metrics(candidates, references)
    assert metrics["structured_laterality_true_positive"] == 1
    assert metrics["structured_laterality_false_positive"] == 1
    assert metrics["structured_laterality_false_negative"] == 1
    assert metrics["structured_laterality_micro_f1"] == pytest.approx(0.5)
    assert metrics["structured_laterality_exact_match"] == pytest.approx(0.5)
