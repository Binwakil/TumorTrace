import pytest

from scripts.evaluate_trust import evaluate_group


def test_trust_group_reports_interval_coverage_and_referral() -> None:
    rows = []
    for index, (dice, uncertainty, target) in enumerate(((0.9, 0.1, 10.0), (0.5, 0.9, 20.0))):
        volumes = {
            region: {
                "value_ml": target,
                "interval_90_ml": [target - 1, target + 1],
                "interval_95_ml": [target - 2, target + 2],
            }
            for region in ("WT", "TC", "ET", "SNFH")
        }
        case = {"macro_dice": dice}
        for region in volumes:
            case[f"target_volume_ml_{region}"] = target
        rows.append(
            {
                "subject_id": str(index),
                "case": case,
                "feature": {
                    "evidence": {
                        "segmentation_uncertainty": uncertainty,
                        "volumes": volumes,
                    }
                },
            }
        )

    result = evaluate_group(rows, random_samples=20)

    assert result["volume_intervals"]["WT"]["0.9"]["coverage"] == 1.0
    assert result["volume_intervals"]["WT"]["0.95"]["mean_width_ml"] == 4.0
    assert result["failure_auroc"] == 1.0
    assert result["referral_at_80pct"]["selective_risk"] == pytest.approx(0.3)
