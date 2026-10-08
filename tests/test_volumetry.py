import numpy as np

from tumortrust_vlm.evaluation.volumetry import (
    aggregate_auxiliary_burden_metrics,
    aggregate_volume_metrics,
    case_volume_metrics,
    concordance_correlation_coefficient,
)


def test_volume_metrics_use_physical_spacing():
    target = np.zeros((10, 10, 10), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[1:3, 1:3, 1:3] = 3
    prediction[1:3, 1:3, 1:2] = 3
    case = case_volume_metrics(prediction, target, (2, 2, 2))
    assert case["target_volume_ml_WT"] == 0.064
    assert case["predicted_volume_ml_WT"] == 0.032
    aggregate = aggregate_volume_metrics([case])
    assert aggregate["median_absolute_volume_error_ml_WT"] == 0.032
    assert concordance_correlation_coefficient(np.array([1, 2]), np.array([1, 2])) == 1.0


def test_absent_regions_use_false_positive_metrics_not_relative_error():
    target = np.zeros((8, 8, 8), dtype=np.uint8)
    prediction = np.zeros_like(target)
    prediction[0, 0, 0] = 3

    case = case_volume_metrics(prediction, target, (1, 1, 1), regions=("ET",))
    aggregate = aggregate_volume_metrics([case], regions=("ET",))

    assert np.isnan(case["relative_volume_error_ET"])
    assert aggregate["absent_region_false_positive_rate_ET"] == 1.0
    assert aggregate["median_false_positive_volume_ml_ET"] == 0.001
    assert np.isnan(aggregate["median_relative_volume_error_ET"])


def test_laterality_aggregation_reports_nonambiguous_f1():
    cases = [
        {
            "predicted_laterality": "left",
            "target_laterality": "left",
            "predicted_volume_ml_WT": 1.0,
            "target_volume_ml_WT": 1.0,
        },
        {
            "predicted_laterality": "right",
            "target_laterality": "right",
            "predicted_volume_ml_WT": 1.0,
            "target_volume_ml_WT": 1.0,
        },
    ]

    aggregate = aggregate_volume_metrics(cases, regions=("WT",))

    assert aggregate["laterality_nonambiguous_cases"] == 2
    assert aggregate["laterality_nonambiguous_f1"] == 1.0


def test_auxiliary_burden_is_compared_with_mask_derived_target() -> None:
    cases = [
        {
            "burden_head_predicted_volume_ml_WT": 8.0,
            "target_volume_ml_WT": 10.0,
        },
        {
            "burden_head_predicted_volume_ml_WT": 24.0,
            "target_volume_ml_WT": 20.0,
        },
    ]

    result = aggregate_auxiliary_burden_metrics(cases, regions=("WT",))

    assert result["burden_head_median_absolute_volume_error_ml_WT"] == 3.0
    assert result["burden_head_median_relative_volume_error_WT"] == 0.2
