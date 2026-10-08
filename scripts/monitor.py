#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.monitor import ProgressRates, collect_status, render_status
from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Live TumorTrust training dashboard")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh period in seconds")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-clear", action="store_true", help="Append instead of clearing")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--json-out",
        help="Optionally update an atomic JSON status file after every refresh",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    rates = ProgressRates()
    try:
        while True:
            status = collect_status(ROOT, rates)
            if args.json_out:
                atomic_json_dump(status, ROOT / args.json_out)
            output = json.dumps(status, indent=2) if args.json else render_status(status, args.interval)
            if not args.once and not args.no_clear:
                print("\033[2J\033[H", end="")
            print(output, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped; training jobs were not interrupted.")


if __name__ == "__main__":
    main()
