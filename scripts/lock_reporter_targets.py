#!/usr/bin/env python
"""T9.7: freeze reporter targets, tokenizer policy, max length, and field schema before any T10
reporter training starts. One-time decision -- must not be revised mid-ablation (PLAN.md's own
rule: "Do not combine loss changes with architecture changes in the same diagnostic run" extends
to changing the target representation mid-comparison too).

What this locks, and why each choice is what it is:

1. Compact-decoder word tokenizer: fit ONLY on report-TRAIN subjects' text (never report-val/test,
   per the existing T2.6/T2.7 lock) using the existing WordTokenizer.fit() defaults
   (minimum_frequency=2, maximum_size=16000) -- already coded in tumortrust_vlm/reporting/dataset.py,
   never previously actually fit+frozen to an artifact. Frozen here.
2. Compact-decoder max_length=256: already the coded default; kept as-is (report word-length
   maximum is 116 per artifacts/report_corpus_audit.json, so 256 word-tokenizer tokens is generous
   headroom even before subword effects, which don't apply to this whole-word tokenizer).
3. Vicuna-path max_length: the current default (192) was never checked against the real LLaMA/Vicuna
   subword tokenizer -- it truncates 8.75% of the 697 report corpus. This script measures the actual
   truncation rate at several candidates and the caller (this run) locks 256, matching the
   compact-decoder value for consistency and cutting truncation to 0.29% (2/697 reports).
4. Field schema: TARGET_SCHEMA_VERSION ("global_finding_nfkc_whitespace_v1", targets.py) and
   FINDING_SCHEMA_VERSION ("brain_mri_normalized_findings_v1", findings.py) are already versioned
   constants in code; this script confirms them as the frozen T10 versions rather than introducing
   new ones.

Writes a public, subject-free lock record to artifacts/reporter_target_lock.json (versions, sizes,
hashes -- no report text or per-subject data) and the frozen vocabulary to
artifacts/private/reporter_word_vocabulary_v1.json (contains no subject IDs, but kept private
alongside this project's other reporting artifacts as a matter of course).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import REPORT_DIRECTORIES
from tumortrust_vlm.data.inventory import load_report_splits
from tumortrust_vlm.reporting.dataset import WordTokenizer
from tumortrust_vlm.reporting.findings import FINDING_SCHEMA_VERSION
from tumortrust_vlm.reporting.targets import TARGET_SCHEMA_VERSION
from tumortrust_vlm.reporting.vicuna_dataset import REPORT_PROMPT
from tumortrust_vlm.utils import atomic_json_dump, sha256_json

VICUNA_MODEL_PATH = "../shared/models/llava-v1.5-7b"
LOCKED_VICUNA_MAX_LENGTH = 256


def report_train_texts(config: dict) -> tuple[list[str], int]:
    manifest_path = ROOT / config["data"]["split_manifest"]
    import json

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]
    eligible = {entry["subject_id"] for entry in entries}
    split_map = load_report_splits(config["data"]["report_split"])

    texts = []
    total_report_train = 0
    for directory in REPORT_DIRECTORIES.values():
        path = Path(config["data"]["report_meta"]) / directory / "global_finding.json"
        reports = json.loads(path.read_text(encoding="utf-8"))
        for subject_id, entry in reports.items():
            if split_map.get(subject_id) != "train":
                continue
            total_report_train += 1
            if subject_id not in eligible:
                continue
            text = entry if isinstance(entry, str) else entry.get("text", str(entry))
            texts.append(text)
    return texts, total_report_train


def measure_vicuna_truncation(config: dict, candidates: list[int]) -> dict:
    import json

    tokenizer = AutoTokenizer.from_pretrained(VICUNA_MODEL_PATH, use_fast=False)
    prompt_len = len(tokenizer.encode(REPORT_PROMPT, add_special_tokens=True))
    lengths = []
    for directory in REPORT_DIRECTORIES.values():
        path = Path(config["data"]["report_meta"]) / directory / "global_finding.json"
        reports = json.loads(path.read_text(encoding="utf-8"))
        for entry in reports.values():
            text = entry if isinstance(entry, str) else entry.get("text", str(entry))
            n = len(tokenizer.encode(" " + text, add_special_tokens=False)) + 1  # +1 eos
            lengths.append(n)
    return {
        "prompt_tokens": prompt_len,
        "reports_measured": len(lengths),
        "truncation_by_max_length": {
            str(cap): sum(1 for n in lengths if n + prompt_len > cap) / len(lengths) for cap in candidates
        },
    }


def main() -> None:
    config = load_config("configs/base.yaml")
    texts, total_report_train = report_train_texts(config)

    tokenizer = WordTokenizer.fit(texts, minimum_frequency=2, maximum_size=16000)
    vocab_path = ROOT / "artifacts/private/reporter_word_vocabulary_v1.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(vocab_path)

    # OOV diagnostic on the same fitting corpus (expected to be low; a held-out check belongs to T10).
    total_tokens = 0
    oov_tokens = 0
    for text in texts:
        for token_id in tokenizer.encode(text, maximum_length=10_000)[1:-1]:  # strip bos/eos
            total_tokens += 1
            if token_id == tokenizer.unk_id:
                oov_tokens += 1

    vicuna_truncation = measure_vicuna_truncation(config, [192, 224, 256, 288])

    lock = {
        "locked_at": "2026-08-17",
        "report_train_subjects": {
            "total_report_train_in_source_split": total_report_train,
            "eligible_after_master_split": len(texts),
        },
        "compact_decoder": {
            "tokenizer": "WordTokenizer (word-level, tumortrust_vlm/reporting/dataset.py)",
            "fit_on": "report-train subjects only (eligible after master split), never val/test",
            "minimum_frequency": 2,
            "maximum_size": 16000,
            "vocabulary_size": len(tokenizer),
            "vocabulary_path": str(vocab_path.relative_to(ROOT)),
            "fitting_corpus_oov_rate": oov_tokens / total_tokens if total_tokens else None,
            "max_length": 256,
            "max_length_rationale": "report word-length maximum is 116 (artifacts/report_corpus_audit.json); "
            "256 word-level tokens is generous headroom.",
        },
        "vicuna_decoder": {
            "tokenizer": f"AutoTokenizer.from_pretrained('{VICUNA_MODEL_PATH}', use_fast=False)",
            "prompt": REPORT_PROMPT,
            "max_length_previous_default": 192,
            "max_length_locked": LOCKED_VICUNA_MAX_LENGTH,
            "truncation_measurement": vicuna_truncation,
            "rationale": "192 truncates 8.75% of the 697-report corpus; 256 (matching the compact "
            "decoder for consistency) cuts that to 0.29% (2 reports).",
        },
        "field_schema": {
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "finding_schema_version": FINDING_SCHEMA_VERSION,
            "status": "confirmed as the frozen T10 versions; already-versioned constants in code, "
            "not newly introduced here",
        },
    }
    lock["lock_sha256"] = sha256_json(lock)
    atomic_json_dump(lock, ROOT / "artifacts/reporter_target_lock.json")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Fitting-corpus OOV rate: {lock['compact_decoder']['fitting_corpus_oov_rate']:.4f}")
    print(f"Vicuna truncation at candidates: {vicuna_truncation['truncation_by_max_length']}")
    print(f"Locked to artifacts/reporter_target_lock.json (sha256 {lock['lock_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
