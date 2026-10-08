from pathlib import Path

from scripts.audit_public_release import audit_root


def test_public_release_audit_accepts_aggregate_artifacts(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "summary.json").write_text(
        '{"subjects": 320, "subject_ids_sha256": "abc", "macro_dice": 0.8457}',
        encoding="utf-8",
    )
    files, violations = audit_root(tmp_path)
    assert len(files) == 1
    assert violations == []


def test_public_release_audit_rejects_structured_subject_and_report_records(
    tmp_path: Path,
) -> None:
    (tmp_path / "cases.jsonl").write_text(
        '{"subject_id": "case-001", "report_text": "example finding"}\n',
        encoding="utf-8",
    )
    _, violations = audit_root(tmp_path)
    assert [(item.rule, item.detail) for item in violations] == [
        ("subject_or_report_record", "report_text, subject_id")
    ]


def test_public_release_audit_rejects_private_paths_weights_and_credentials(
    tmp_path: Path,
) -> None:
    private = tmp_path / "artifacts" / "private"
    private.mkdir(parents=True)
    (private / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.ckpt").write_bytes(b"checkpoint")
    fake_key = "sk" + "-" + "A" * 24
    (tmp_path / "notes.txt").write_text(fake_key, encoding="utf-8")
    _, violations = audit_root(tmp_path)
    assert {(item.path, item.rule) for item in violations} == {
        ("artifacts/private/manifest.json", "restricted_path"),
        ("model.ckpt", "restricted_path"),
        ("notes.txt", "api_credential"),
    }
