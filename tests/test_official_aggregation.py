from scripts.evaluate_official_brats import aggregate_results


def test_official_aggregation_averages_region_metrics_and_skips_subject_name():
    records = [
        {"wt": {"dice": 0.5, "tp": 1}, "subject_name": "a"},
        {"wt": {"dice": 1.0, "tp": 2}, "subject_name": "b"},
    ]

    result = aggregate_results(records)

    assert result["wt"]["dice"] == 0.75
    assert result["wt"]["tp"] == 1.5
