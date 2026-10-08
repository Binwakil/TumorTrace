import torch

from tumortrust_vlm.engine import evaluate_model
from tumortrust_vlm.inference import infer_core_once
from tumortrust_vlm.models.core import MultiTaskSegResNet, SeparateEncoderMultiTaskSegResNet
from tumortrust_vlm.models.losses import MultiTaskObjective, pcgrad_backward
from tumortrust_vlm.models.reporter import EvidenceConditionedReporter, RegionAwarePooler


def test_parallel_core_shapes_and_gradients():
    model = MultiTaskSegResNet(init_filters=8, dropout=0.0)
    image = torch.randn(1, 4, 32, 32, 32)
    output = model(image)
    assert output["segmentation"].shape == (1, 4, 32, 32, 32)
    assert output["classification"].shape == (1, 3)
    loss = MultiTaskObjective()(
        output, torch.zeros(1, 32, 32, 32, dtype=torch.long), torch.tensor([0])
    )
    loss["loss"].backward()
    assert model.backbone.convInit.conv.weight.grad is not None
    assert model.classifier[-1].weight.grad is not None


def test_classification_only_objective_does_not_require_segmentation_logits():
    objective = MultiTaskObjective(mode="classification_only")
    outputs = {"classification": torch.randn(2, 3, requires_grad=True)}
    losses = objective(outputs, torch.zeros(2, 8, 8, 8, dtype=torch.long), torch.tensor([0, 2]))
    losses["loss"].backward()

    assert outputs["classification"].grad is not None
    assert losses["segmentation_loss"].item() == 0.0


def test_separate_encoder_control_has_disjoint_backbone_gradients():
    model = SeparateEncoderMultiTaskSegResNet(init_filters=8, dropout=0.0)
    output = model(torch.randn(1, 4, 32, 32, 32))
    output["classification"].sum().backward()

    assert model.classification_backbone.convInit.conv.weight.grad is not None
    assert model.backbone.convInit.conv.weight.grad is None


def test_core_inference_uses_sliding_window_and_returns_global_features():
    model = MultiTaskSegResNet(init_filters=8, dropout=0.0).eval()
    image = torch.randn(1, 4, 32, 32, 48)
    presence = torch.ones(1, 4)

    with torch.no_grad():
        segmentation, classification, latent = infer_core_once(
            model,
            image,
            presence,
            roi_size=(32, 32, 32),
            overlap=0.25,
        )

    assert segmentation.shape == (1, 4, 32, 32, 48)
    assert classification.shape == (1, 3)
    assert latent.shape[:2] == (1, 64)


def test_segmentation_only_evaluation_skips_untrained_classifier():
    model = MultiTaskSegResNet(init_filters=8, dropout=0.0).eval()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("untrained classification head was evaluated")

    model.classifier.forward = fail_if_called
    batch = {
        "image": torch.randn(1, 4, 32, 32, 32),
        "label": torch.zeros(1, 32, 32, 32, dtype=torch.long),
        "class_label": torch.tensor([0]),
        "modality_presence": torch.ones(1, 4),
        "subject_id": ["subject"],
        "cohort": ["GLI"],
        "source_branch": ["source"],
        "source_orientation": ["RAS"],
        "has_report": [False],
        "spacing": torch.ones(1, 3),
    }
    config = {
        "data": {"patch_size": [32, 32, 32]},
        "model": {"joint_weighting": "segmentation_only"},
        "inference": {"sliding_window_overlap": 0.5, "sw_batch_size": 1},
        "evaluation": {"regions": ["WT"], "component_min_volume_mm3": 100},
    }

    aggregate, cases = evaluate_model(
        model, [batch], config, torch.device("cpu"), patch_validation=True
    )

    assert aggregate["classification_evaluated"] is False
    assert "balanced_accuracy" not in aggregate
    assert "class_probability" not in cases[0]


def test_pcgrad_projects_conflicting_shared_gradients():
    shared = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    task_specific = torch.nn.Parameter(torch.tensor([1.0]))
    model = torch.nn.ParameterList([shared, task_specific])
    segmentation = shared[0] + shared[1] + task_specific[0]
    classification = -shared[0] - shared[1]

    result = pcgrad_backward(model, segmentation, classification)

    assert result["pcgrad_conflict"] == 1.0
    assert shared.grad is not None
    assert task_specific.grad is not None
    assert task_specific.grad.item() == 1.0
    assert torch.isfinite(shared.grad).all()


def test_reporter_uses_visual_and_evidence_paths():
    pooler = RegionAwarePooler()
    feature = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    regions = torch.softmax(torch.randn(2, 4, 16, 16, 16), dim=1)
    tokens = pooler(feature, regions)
    reporter = EvidenceConditionedReporter(40, 16, 10, hidden_dim=32, embedding_dim=16)
    logits = reporter(
        torch.ones(2, 6, dtype=torch.long), visual_tokens=tokens, evidence=torch.randn(2, 10)
    )
    assert logits.shape == (2, 6, 40)
    logits.sum().backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert reporter.evidence_adapter[0].weight.grad is not None

    reconstructed = reporter.reconstruct_fields(tokens.detach(), torch.randn(2, 10), 2)
    assert reconstructed.shape == (2, 10)
    text_only = reporter.generate(
        1,
        2,
        4,
        visual_tokens=None,
        evidence=None,
        batch_size=2,
    )
    assert text_only.shape[0] == 2
