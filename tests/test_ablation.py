from pathlib import Path

import pytest
import yaml

from tumortrust_vlm.ablation import (
    changed_config_fields,
    materialize_ablation_configs,
    run_ablation_configs,
)


def test_conditional_ablation_requires_explicit_trigger(tmp_path: Path) -> None:
    base = {
        "project": {"name": "test", "output_dir": "outputs/base"},
        "model": {"architecture": "shared"},
    }
    matrix = {
        "experiments": [
            {
                "id": "conditional",
                "operation": "train",
                "trigger": "only_if_negative_transfer",
                "overrides": {"model.architecture": "separate"},
            }
        ]
    }
    base_path = tmp_path / "base.yaml"
    matrix_path = tmp_path / "matrix.yaml"
    base_path.write_text(yaml.safe_dump(base))
    matrix_path.write_text(yaml.safe_dump(matrix))
    paths = materialize_ablation_configs(str(base_path), str(matrix_path), str(tmp_path / "out"))

    skipped = run_ablation_configs(paths, "python", dry_run=True)
    enabled = run_ablation_configs(
        paths,
        "python",
        dry_run=True,
        enabled_triggers={"only_if_negative_transfer"},
    )

    assert skipped[0]["status"] == "skipped_conditional_trigger_not_enabled"
    assert enabled[0]["status"] == "dry_run"
    assert enabled[0]["command"][-2:] == ["--config", str(paths[0])]


def test_changed_config_fields_ignores_run_metadata() -> None:
    baseline = {
        "project": {"name": "baseline", "output_dir": "outputs/baseline"},
        "data": {"normalization": "robust", "missing_modality_probability": 0.0},
    }
    candidate = {
        "project": {"name": "candidate", "output_dir": "outputs/candidate"},
        "data": {"normalization": "zscore", "missing_modality_probability": 0.0},
    }

    assert changed_config_fields(candidate, baseline) == {"data.normalization"}


def test_materialization_rejects_undeclared_ablation_confounds(tmp_path: Path) -> None:
    base = {
        "project": {"name": "test", "output_dir": "outputs/base"},
        "data": {"normalization": "robust", "missing_modality_probability": 0.0},
        "model": {"joint_weighting": "fixed"},
    }
    matrix = {
        "experiments": [
            {
                "id": "control",
                "overrides": {"model.joint_weighting": "segmentation_only"},
            },
            {
                "id": "confounded",
                "compare_to": "control",
                "expected_changed_fields": ["data.normalization"],
                "overrides": {
                    "data.normalization": "zscore",
                    "data.missing_modality_probability": 0.15,
                    "model.joint_weighting": "segmentation_only",
                },
            },
        ]
    }
    base_path = tmp_path / "base.yaml"
    matrix_path = tmp_path / "matrix.yaml"
    base_path.write_text(yaml.safe_dump(base))
    matrix_path.write_text(yaml.safe_dump(matrix))

    with pytest.raises(ValueError, match="violates single-variable declaration"):
        materialize_ablation_configs(str(base_path), str(matrix_path), str(tmp_path / "out"))


def test_project_ablation_matrix_has_valid_declared_comparisons(tmp_path: Path) -> None:
    paths = materialize_ablation_configs(
        "configs/base.yaml", "configs/ablations.yaml", str(tmp_path / "resolved")
    )

    assert len(paths) == 20
    zscore = yaml.safe_load((tmp_path / "resolved" / "D3_zscore.yaml").read_text())
    modality_dropout = yaml.safe_load(
        (tmp_path / "resolved" / "D4_with_modality_dropout.yaml").read_text()
    )
    assert zscore["model"]["joint_weighting"] == "segmentation_only"
    assert zscore["data"]["missing_modality_probability"] == 0.0
    assert modality_dropout["model"]["joint_weighting"] == "segmentation_only"
    assert modality_dropout["data"]["missing_modality_probability"] == 0.15
