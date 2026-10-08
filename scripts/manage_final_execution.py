#!/usr/bin/env python
"""Verify the frozen protocol and maintain the irreversible final-test execution receipt."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.utils import atomic_json_dump, sha256_file

RECEIPT = ROOT / "artifacts/private/final_test_execution_receipt.json"


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("begin", "complete", "fail"))
    parser.add_argument("--protocol", default="artifacts/final_protocol_lock.json")
    parser.add_argument("--message")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    if not protocol_path.is_file():
        raise SystemExit("Final protocol lock is missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise SystemExit("Final protocol is not frozen")

    if args.action == "begin":
        if RECEIPT.exists():
            existing = json.loads(RECEIPT.read_text(encoding="utf-8"))
            raise SystemExit(
                f"Final test has already been invoked (status={existing.get('status')}); refusing rerun"
            )
        mismatches = []
        for relative, expected in protocol["files"].items():
            path = ROOT / relative
            actual = sha256_file(path) if path.is_file() else None
            if actual != expected["sha256"]:
                mismatches.append(relative)
        if mismatches:
            raise SystemExit(f"Frozen input hash mismatch: {mismatches}")
        receipt = {
            "status": "running",
            "started_at": now(),
            "protocol": args.protocol,
            "protocol_sha256": sha256_file(protocol_path),
            "locked_test_opened": True,
        }
    else:
        if not RECEIPT.is_file():
            raise SystemExit("Cannot update a final execution that was never begun")
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if receipt.get("status") != "running":
            raise SystemExit(f"Final execution is already {receipt.get('status')}")
        receipt["status"] = "complete" if args.action == "complete" else "failed"
        receipt["finished_at"] = now()
        receipt["message"] = args.message
        summary = ROOT / "artifacts/locked_final_summary.json"
        if args.action == "complete":
            if not summary.is_file():
                raise SystemExit("Cannot complete final execution without the locked-final summary")
            receipt["summary"] = str(summary.relative_to(ROOT))
            receipt["summary_sha256"] = sha256_file(summary)
    atomic_json_dump(receipt, RECEIPT)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
