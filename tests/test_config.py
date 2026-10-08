import json

import numpy as np

from tumortrust_vlm.config import config_hash, with_overrides
from tumortrust_vlm.utils import atomic_json_dump


def test_overrides_do_not_mutate_source():
    source = {"model": {"dropout": 0.1}}
    changed = with_overrides(source, {"model.dropout": 0.2, "training.epochs": 3})
    assert source == {"model": {"dropout": 0.1}}
    assert changed["model"]["dropout"] == 0.2
    assert changed["training"]["epochs"] == 3
    assert config_hash(source) == config_hash({"model": {"dropout": 0.1}})


def test_atomic_json_dump_serializes_nonfinite_values_as_null(tmp_path):
    output = tmp_path / "result.json"
    atomic_json_dump(
        {"nan": float("nan"), "inf": np.float64("inf"), "value": np.int64(3)},
        output,
    )

    assert json.loads(output.read_text()) == {"nan": None, "inf": None, "value": 3}
    assert "NaN" not in output.read_text()
    assert "Infinity" not in output.read_text()
