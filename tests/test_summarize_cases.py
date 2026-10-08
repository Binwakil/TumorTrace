import importlib.util
from pathlib import Path


def load_summarize():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_cases.py"
    spec = importlib.util.spec_from_file_location("summarize_cases", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def test_summarize_segmentation_cases_omits_classification_metrics():
    summarize = load_summarize()
    case = {
        "subject_id": "case",
        "cohort": "GLI",
        "source_branch": "source",
        "macro_dice": 1.0,
        "dice_WT": 1.0,
        "predicted_volume_ml_WT": 1.0,
        "target_volume_ml_WT": 1.0,
        "predicted_laterality": "left",
        "target_laterality": "left",
        "laterality_correct": 1.0,
        "target_component_count_WT": 1.0,
    }

    result = summarize([case])

    assert result["subjects"] == 1
    assert result["classification_evaluated"] is False
    assert result["laterality_nonambiguous_f1"] == 0.5
