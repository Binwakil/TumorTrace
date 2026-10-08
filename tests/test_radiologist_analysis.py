from tumortrust_vlm.evaluation.radiologist import (
    fleiss_kappa,
    summarize_radiologist_scores,
)


def test_fleiss_kappa_is_one_for_perfect_agreement():
    assert fleiss_kappa([[True, True], [False, False]]) == 1.0


def test_radiologist_summary_applies_safety_gates():
    private_map = [
        {
            "blinded_report_id": "x",
            "subject_id": "case",
            "method": "R0",
            "session": 1,
        }
    ]
    score = {
        "blinded_report_id": "x",
        "reader_id": "reader",
        "factual_correctness": "no_error",
        "completeness": "no_important_omission",
        "clinically_significant_error_or_omission": False,
        "potentially_harmful_error": False,
        "unsupported_statement": False,
        "quantitative_contradiction": False,
        "laterality_contradiction": False,
        "usefulness_1_to_5": 5,
        "required_editing": "none",
    }

    result = summarize_radiologist_scores([score], private_map, bootstrap_samples=20)

    assert result["methods"]["R0"]["passes_no_significant_error_gate"]
    assert result["methods"]["R0"]["passes_harmful_error_gate"]
    assert result["agreement"]["potentially_harmful_error"]["kappa"] is None
