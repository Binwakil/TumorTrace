#!/usr/bin/env python
"""Fail a proposed public release when restricted research artifacts are present.

By default the audit examines every Git-tracked and nonignored file in the repository, not only
the current diff. Passing ``--root`` also permits the same gate to be run on a staged release tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RESTRICTED_PREFIXES = (
    ("artifacts", "private"),
    ("checkpoints",),
    ("data", "processed"),
    ("data", "raw"),
    ("outputs",),
    ("reports", "private"),
    ("results", "predictions"),
)
RESTRICTED_SUFFIXES = (
    ".ckpt",
    ".nii",
    ".nii.gz",
    ".npz",
    ".pt",
    ".pth",
)
STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl", ".tsv"}
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".csv",
    ".env",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
SENSITIVE_RECORD_KEYS = {
    "generated_report",
    "normalized_reference_findings",
    "reference_report",
    "report_text",
    "source_subject_identifier",
    "subject_id",
}
SECRET_PATTERN = re.compile(r"\bsk" + r"-(?:proj-)?[A-Za-z0-9_-]{20,}\b")


@dataclass(frozen=True)
class Violation:
    path: str
    rule: str
    detail: str


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def discover_files(root: Path) -> list[Path]:
    """Return all public-committable files, or all files in a non-Git staging tree."""
    root = root.resolve()
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode == 0:
        paths = []
        for raw in completed.stdout.split(b"\0"):
            if not raw:
                continue
            candidate = (root / raw.decode("utf-8", errors="surrogateescape")).resolve()
            if candidate.is_file() and _is_relative_to(candidate, root):
                paths.append(candidate)
        return sorted(set(paths))
    return sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)


def _restricted_path(relative: Path) -> str | None:
    parts = relative.parts
    for prefix in RESTRICTED_PREFIXES:
        if parts[: len(prefix)] == prefix:
            return "/".join(prefix)
    lower_name = relative.name.lower()
    for suffix in RESTRICTED_SUFFIXES:
        if lower_name.endswith(suffix):
            return suffix
    return None


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def _structured_keys(path: Path, text: str) -> set[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return set(_walk_keys(json.loads(text)))
        except json.JSONDecodeError:
            return set()
    if suffix == ".jsonl":
        keys: set[str] = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                keys.update(_walk_keys(json.loads(line)))
            except json.JSONDecodeError:
                continue
        return keys
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        return set(next(reader, []))
    return set()


def audit_file(root: Path, path: Path) -> list[Violation]:
    relative = path.resolve().relative_to(root.resolve())
    display = relative.as_posix()
    violations: list[Violation] = []
    restricted = _restricted_path(relative)
    if restricted is not None:
        violations.append(Violation(display, "restricted_path", restricted))
        return violations
    if path.is_symlink():
        violations.append(Violation(display, "symlink", "release files must be regular files"))
        return violations
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return violations
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return [Violation(display, "unreadable", str(error))]
    if SECRET_PATTERN.search(text):
        violations.append(Violation(display, "api_credential", "OpenAI-style secret token"))
    if path.suffix.lower() in STRUCTURED_SUFFIXES:
        exposed = sorted(_structured_keys(path, text) & SENSITIVE_RECORD_KEYS)
        if exposed:
            violations.append(
                Violation(display, "subject_or_report_record", ", ".join(exposed))
            )
    return violations


def audit_root(root: Path) -> tuple[list[Path], list[Violation]]:
    files = discover_files(root)
    violations = [violation for path in files for violation in audit_file(root, path)]
    return files, sorted(violations, key=lambda item: (item.path, item.rule, item.detail))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    files, violations = audit_root(root)
    result = {
        "root": str(root),
        "files_scanned": len(files),
        "status": "pass" if not violations else "fail",
        "violations": [asdict(violation) for violation in violations],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
