import torch

from scripts.extract_core_features import checkpoint_config_differences
from tumortrust_vlm.inference import crop_to_shape, pad_to_multiple


def test_padding_round_trip():
    image = torch.randn(1, 4, 31, 33, 35)
    padded, shape = pad_to_multiple(image, 16)
    assert padded.shape[-3:] == (32, 48, 48)
    assert torch.equal(crop_to_shape(padded, shape), image)


def test_checkpoint_config_difference_ignores_numeric_weight_decay_serialization():
    reference = {
        "project": {"seed": 1},
        "training": {"weight_decay": "1e-05", "learning_rate": 2e-4},
    }
    candidate = {
        "project": {"seed": 2},
        "training": {"weight_decay": 1.0e-5, "learning_rate": 2e-4},
    }

    assert checkpoint_config_differences(reference, candidate) == {"project.seed"}
