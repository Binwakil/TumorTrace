import json

import numpy as np
import pytest
import torch

from scripts.evaluate_burden_head import main as evaluate_main
from scripts.evaluate_burden_head import summarize
from scripts.train_burden_head import assemble
from scripts.train_burden_head import main as train_main
from tumortrust_vlm.models.burden import FrozenFeatureBurdenMLP


def test_frozen_feature_burden_model_shape() -> None:
    model = FrozenFeatureBurdenMLP(8, hidden_dim=16, dropout=0.0)
    assert model(torch.randn(3, 8)).shape == (3, 4)


def test_burden_assembly_requires_exact_subject_pairing(tmp_path) -> None:
    features = tmp_path / "features.json"
    evaluation = tmp_path / "evaluation.json"
    features.write_text(
        json.dumps(
            [
                {
                    "subject_id": "a",
                    "global_features": [1.0, 2.0],
                    "inference_provenance": {"checkpoint_sha256": ["abc"]},
                }
            ]
        )
    )
    evaluation.write_text(
        json.dumps(
            [
                {
                    "subject_id": "b",
                    **{f"target_volume_ml_{region}": 1.0 for region in ("WT", "TC", "ET", "SNFH")},
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="subject mismatch"):
        assemble(str(features), str(evaluation))


def test_burden_assembly_rejects_duplicate_subjects(tmp_path) -> None:
    row = {
        "subject_id": "a",
        "global_features": [1.0, 2.0],
        "inference_provenance": {"checkpoint_sha256": ["abc"]},
    }
    features = tmp_path / "features.json"
    evaluation = tmp_path / "evaluation.json"
    features.write_text(json.dumps([row, row]))
    evaluation.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="Duplicate subject_id"):
        assemble(str(features), str(evaluation))


def test_burden_summary_uses_correct_deterministic_baseline() -> None:
    case = {
        "predicted_laterality": "left",
        "target_laterality": "left",
    }
    for region in ("WT", "TC", "ET", "SNFH"):
        case[f"target_volume_ml_{region}"] = 10.0
        case[f"predicted_volume_ml_{region}"] = 9.0
        case[f"burden_head_predicted_volume_ml_{region}"] = 7.0
    result = summarize([case])
    wt = result["comparison"]["WT"]
    assert wt["deterministic_median_absolute_error_ml"] == 1.0
    assert wt["learned_median_absolute_error_ml"] == 3.0
    assert wt["learned_minus_deterministic_absolute_error_ml"] == 2.0
    assert np.isfinite(wt["learned_ccc"])


def test_burden_training_resume_is_exact_and_evaluation_is_traceable(
    tmp_path, monkeypatch
) -> None:
    provenance = {
        "ensemble_checkpoints": 1,
        "checkpoint_sha256": ["frozen-core"],
        "mc_samples_per_checkpoint": 1,
    }

    def records(prefix: str, count: int) -> tuple[list[dict], list[dict]]:
        features = []
        evaluations = []
        for index in range(count):
            subject_id = f"{prefix}-{index}"
            feature = [float(index), float(index % 2), float(index**2), 1.0]
            features.append(
                {
                    "subject_id": subject_id,
                    "global_features": feature,
                    "inference_provenance": provenance,
                }
            )
            evaluation = {
                "subject_id": subject_id,
                "cohort": ("GLI", "MEN", "MET")[index % 3],
                "source_branch": "synthetic",
                "predicted_laterality": "left",
                "target_laterality": "left",
            }
            for region_index, region in enumerate(("WT", "TC", "ET", "SNFH"), start=1):
                target = float(index + region_index)
                evaluation[f"target_volume_ml_{region}"] = target
                evaluation[f"predicted_volume_ml_{region}"] = target - 0.5
            evaluations.append(evaluation)
        return features, evaluations

    paths = {}
    for split, count in (("train", 8), ("val", 4)):
        features, evaluations = records(split, count)
        paths[f"{split}_features"] = tmp_path / f"{split}_features.json"
        paths[f"{split}_evaluation"] = tmp_path / f"{split}_evaluation.json"
        paths[f"{split}_features"].write_text(json.dumps(features))
        paths[f"{split}_evaluation"].write_text(json.dumps(evaluations))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def train(output, epochs: int, resume=None) -> None:
        arguments = [
            "train_burden_head.py",
            "--train-features",
            str(paths["train_features"]),
            "--train-evaluation",
            str(paths["train_evaluation"]),
            "--val-features",
            str(paths["val_features"]),
            "--val-evaluation",
            str(paths["val_evaluation"]),
            "--output",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            "4",
            "--hidden-dim",
            "8",
            "--dropout",
            "0.1",
            "--patience",
            "10",
        ]
        if resume:
            arguments.extend(("--resume", str(resume)))
        monkeypatch.setattr("sys.argv", arguments)
        train_main()

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    train(uninterrupted, 4)
    train(resumed, 2)
    train(resumed, 4, resumed / "last.pt")
    full_checkpoint = torch.load(uninterrupted / "last.pt", weights_only=False)
    resumed_checkpoint = torch.load(resumed / "last.pt", weights_only=False)
    for name, tensor in full_checkpoint["model"].items():
        assert torch.equal(tensor, resumed_checkpoint["model"][name])

    summary_path = tmp_path / "burden_summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_burden_head.py",
            "--features",
            str(paths["val_features"]),
            "--evaluation",
            str(paths["val_evaluation"]),
            "--checkpoint",
            str(resumed / "best.pt"),
            "--output",
            str(summary_path),
        ],
    )
    evaluate_main()
    summary = json.loads(summary_path.read_text())
    assert summary["overall"]["subjects"] == 4
    assert summary["feature_inference_provenance"] == provenance
