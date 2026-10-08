from scripts.materialize_cohort_models import cohort_manifest


def test_cohort_manifest_excludes_other_cohorts_and_final_test() -> None:
    master = {
        "entries": [
            {"subject_id": f"{cohort}-{split}", "cohort": cohort, "split": split}
            for cohort in ("GLI", "MEN", "MET")
            for split in ("train", "val", "test")
        ]
    }
    result = cohort_manifest(master, "MEN")

    assert {entry["cohort"] for entry in result["entries"]} == {"MEN"}
    assert {entry["split"] for entry in result["entries"]} == {"train", "val"}
    assert not result["contains_source_final_test"]
    assert not result["final_test_locked"]
