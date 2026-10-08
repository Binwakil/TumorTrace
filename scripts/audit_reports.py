#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.config import load_config
from tumortrust_vlm.data.constants import REPORT_DIRECTORIES
from tumortrust_vlm.data.inventory import load_report_splits
from tumortrust_vlm.evaluation.reporting import report_diversity
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="+", default=("train", "val"), choices=("train", "val", "test"))
    args = parser.parse_args()
    if "test" in args.splits:
        raise SystemExit("Test reports remain locked during corpus development")
    config = load_config(args.config)
    split_map = load_report_splits(config["data"]["report_split"])
    texts = []
    cohort_counts = Counter()
    for cohort, directory in REPORT_DIRECTORIES.items():
        path = Path(config["data"]["report_meta"]) / directory / "global_finding.json"
        reports = json.loads(path.read_text(encoding="utf-8"))
        for subject_id, text in reports.items():
            if split_map.get(subject_id) in args.splits:
                texts.append(text)
                cohort_counts[cohort] += 1
    token_lengths = [len(re.findall(r"\S+", text)) for text in texts]
    summary = {
        "splits": list(args.splits),
        "subjects": len(texts),
        "cohorts": dict(cohort_counts),
        "empty_reports": sum(not text.strip() for text in texts),
        "word_length_median": sorted(token_lengths)[len(token_lengths) // 2],
        "word_length_maximum": max(token_lengths),
        **report_diversity(texts),
    }
    atomic_json_dump(summary, ROOT / "artifacts" / "report_corpus_audit.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

