#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

RATING_TEMPLATE = {
    "factual_correctness": None,
    "completeness": None,
    "clinically_significant_error_or_omission": None,
    "potentially_harmful_error": None,
    "unsupported_statement": None,
    "unsupported_text_span": None,
    "quantitative_contradiction": None,
    "laterality_contradiction": None,
    "usefulness_1_to_5": None,
    "required_editing": None,
}


def _report_text(record: dict) -> str:
    text = record.get("report", record.get("prediction"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Every radiologist record requires a non-empty report/prediction")
    return text


def build_packets(
    records: list[dict],
    *,
    per_stratum: int,
    seed: int,
    requested_methods: tuple[str, ...] | None = None,
) -> tuple[list[list[dict]], list[dict], dict]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    by_subject: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        subject_id = record["subject_id"]
        method = record["method"]
        if method in by_subject[subject_id]:
            raise ValueError(f"Duplicate subject/method record: {subject_id}/{method}")
        by_subject[subject_id][method] = record
    methods = (
        tuple(requested_methods)
        if requested_methods
        else tuple(sorted({record["method"] for record in records}))
    )
    if len(methods) < 2:
        raise ValueError("At least two report methods are required for blinded comparison")
    eligible = {}
    incomplete = []
    for subject_id, method_records in by_subject.items():
        missing = set(methods) - set(method_records)
        if missing:
            incomplete.append(subject_id)
            continue
        reference = method_records[methods[0]]
        metadata = {
            "cohort": reference["cohort"],
            "burden_quartile": reference["burden_quartile"],
            "lesion_count": reference.get("lesion_count", "unspecified"),
            "referral": bool(reference["referral"]),
            "error_quartile": reference.get("error_quartile", "unspecified"),
        }
        for method in methods[1:]:
            candidate = method_records[method]
            for key, value in metadata.items():
                candidate_value = (
                    bool(candidate[key]) if key == "referral" else candidate.get(key, "unspecified")
                )
                if candidate_value != value:
                    raise ValueError(f"Inconsistent {key} across methods for subject {subject_id}")
        eligible[subject_id] = {"metadata": metadata, "records": method_records}
    if incomplete:
        raise ValueError(f"{len(incomplete)} subjects lack one or more requested report methods")

    strata: dict[tuple, list[str]] = defaultdict(list)
    for subject_id, payload in eligible.items():
        metadata = payload["metadata"]
        strata[
            (
                metadata["cohort"],
                metadata["burden_quartile"],
                metadata["lesion_count"],
                metadata["referral"],
                metadata["error_quartile"],
            )
        ].append(subject_id)
    rng = random.Random(seed)
    selected = []
    for stratum in sorted(strata):
        values = sorted(strata[stratum])
        rng.shuffle(values)
        selected.extend(values[:per_stratum])
    rng.shuffle(selected)

    method_order = list(methods)
    rng.shuffle(method_order)
    packets: list[list[dict]] = [[] for _ in methods]
    private_map = []
    report_counter = 1
    for subject_index, subject_id in enumerate(selected):
        for session_index in range(len(methods)):
            method = method_order[(subject_index + session_index) % len(methods)]
            record = eligible[subject_id]["records"][method]
            blinded_report_id = f"TT-RAD-{report_counter:05d}"
            report_counter += 1
            packets[session_index].append(
                {
                    "blinded_report_id": blinded_report_id,
                    "report": _report_text(record),
                    "ratings": dict(RATING_TEMPLATE),
                }
            )
            private_map.append(
                {
                    "blinded_report_id": blinded_report_id,
                    "session": session_index + 1,
                    "subject_id": subject_id,
                    "method": method,
                }
            )
    for packet in packets:
        rng.shuffle(packet)
    summary = {
        "subjects": len(selected),
        "methods": list(methods),
        "sessions": len(packets),
        "reports": len(private_map),
        "strata": len(strata),
        "same_subject_reports_per_session": 1,
    }
    return packets, private_map, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        required=True,
        help=(
            "JSON records with subject_id, method, cohort, report/prediction, "
            "burden_quartile, lesion_count, referral, and error_quartile"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()
    if not args.unlock_final_test:
        raise SystemExit(
            "Radiologist packets are generated only after the final protocol is frozen"
        )
    records = json.loads(Path(args.reports).read_text(encoding="utf-8"))
    packets, private_map, summary = build_packets(
        records,
        per_stratum=args.per_stratum,
        seed=args.seed,
        requested_methods=tuple(args.methods) if args.methods else None,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, packet in enumerate(packets, start=1):
        (output / f"review_packet_session_{index}.json").write_text(
            json.dumps(packet, indent=2) + "\n", encoding="utf-8"
        )
    (output / "private_randomization_map.json").write_text(
        json.dumps(private_map, indent=2) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(json.dumps(private_map, sort_keys=True).encode()).hexdigest()
    (output / "randomization_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({**summary, "randomization_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
