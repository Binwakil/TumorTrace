from tumortrust_vlm.data.splits import (
    assert_zero_overlap,
    build_master_split,
    split_requires_final_unlock,
)


def record(index: int, cohort: str, report_split=None):
    return {
        "subject_id": f"BraTS-{cohort}-{index:05d}-000",
        "cohort": cohort,
        "source_branch": "TrainingData",
        "labeled": True,
        "complete_images": True,
        "has_report": report_split is not None,
        "report_split": report_split,
        "identity_fingerprint": f"fingerprint-{cohort}-{index}",
    }


def test_locked_report_membership_and_no_overlap():
    records = []
    for cohort in ("GLI", "MEN", "MET"):
        records.extend(record(index, cohort) for index in range(20))
    records[0]["has_report"] = True
    records[0]["report_split"] = "test"
    manifest = build_master_split(records, seed=3)
    assert_zero_overlap(manifest)
    entry = next(
        item for item in manifest["entries"] if item["subject_id"] == records[0]["subject_id"]
    )
    assert entry["split"] == "test"
    assert entry["split_reason"] == "locked_report_split"
    assert manifest["final_test_locked"] is True


def test_split_is_deterministic():
    records = [record(index, "GLI") for index in range(30)]
    assert (
        build_master_split(records, seed=9)["manifest_sha256"]
        == build_master_split(records, seed=9)["manifest_sha256"]
    )


def test_master_test_role_requires_final_unlock():
    manifest = {"final_test_locked": True}
    assert split_requires_final_unlock(manifest, "test")
    assert not split_requires_final_unlock(manifest, "val")


def test_development_cross_evaluation_test_role_does_not_require_unlock():
    manifest = {
        "contains_source_final_test": False,
        "final_test_locked": False,
        "source_evaluation_split": "val",
    }
    assert not split_requires_final_unlock(manifest, "test")
