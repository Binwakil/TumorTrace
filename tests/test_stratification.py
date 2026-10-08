from tumortrust_vlm.evaluation.stratification import stratified_case_metrics


def test_stratification_reports_cohort_source_burden_and_lesion_count() -> None:
    cases = [
        {
            "cohort": cohort,
            "source_branch": source,
            "macro_dice": 0.7 + index / 100,
            "target_volume_ml_WT": float(index + 1),
            "target_component_count_WT": float(1 if index < 2 else 2),
        }
        for index, (cohort, source) in enumerate(
            (("GLI", "A"), ("GLI", "A"), ("MEN", "B"), ("MET", "C"))
        )
    ]

    result = stratified_case_metrics(cases)

    assert set(result["cohort"]) == {"GLI", "MEN", "MET"}
    assert result["source"]["GLI:A"]["subjects"] == 2
    assert set(result["lesion_count"]) == {"single", "multifocal"}
    assert sum(group["subjects"] for group in result["burden_quartile"].values()) == 4
