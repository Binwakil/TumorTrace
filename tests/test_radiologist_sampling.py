from scripts.sample_radiologist_cases import build_packets


def test_radiologist_packets_separate_methods_for_each_subject_by_session():
    records = []
    for subject in ("a", "b"):
        for method in ("R0", "R2", "R4"):
            records.append(
                {
                    "subject_id": subject,
                    "method": method,
                    "cohort": "GLI",
                    "burden_quartile": "Q1",
                    "lesion_count": "single",
                    "referral": False,
                    "error_quartile": "Q1",
                    "report": f"report {subject} {method}",
                }
            )

    packets, private_map, summary = build_packets(records, per_stratum=5, seed=7)

    assert len(packets) == 3
    assert all(len(packet) == 2 for packet in packets)
    assert summary["reports"] == 6
    for subject in ("a", "b"):
        assignments = [row for row in private_map if row["subject_id"] == subject]
        assert {row["method"] for row in assignments} == {"R0", "R2", "R4"}
        assert {row["session"] for row in assignments} == {1, 2, 3}
