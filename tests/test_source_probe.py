import importlib.util
from pathlib import Path

import numpy as np


def load_probe_function():
    path = Path(__file__).resolve().parents[1] / "scripts" / "source_probe.py"
    spec = importlib.util.spec_from_file_location("source_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe_target


def test_source_probe_handles_single_class_without_crashing():
    probe = load_probe_function()
    result = probe(np.ones((6, 3)), np.zeros(6, dtype=int), "source")

    assert result["source_status"] == "infeasible_single_class"
    assert result["source_balanced_accuracy"] is None


def test_source_probe_adapts_folds_to_smallest_class():
    probe = load_probe_function()
    rng = np.random.default_rng(4)
    features = rng.normal(size=(10, 3))
    target = np.asarray([0] * 3 + [1] * 7)
    result = probe(features, target, "source")

    assert result["source_status"] == "complete"
    assert result["source_cv_folds"] == 3
