#!/usr/bin/env python
"""Materialize a config that changes ONLY a whitelisted set of fields from a frozen base.

Used to build "matched" comparator runs (e.g. a MET source-holdout classifier that is identical to
the frozen C0 classifier except for its split manifest, or a separate-encoder control identical to
M0 except for its architecture) without silently drifting any other locked training/model parameter.

`project.name`, `project.output_dir`, `project.operation`, and `project.trigger` are always
permitted to change (matching `tumortrust_vlm.ablation.changed_config_fields`'s existing ignore
list) since every run needs its own identity/output location. Every other field must either be
unchanged from `--base` or explicitly listed in `--allow-changed`; any other drift fails loudly
before a config is written, let alone before GPU time is spent training it.

Example:
    python scripts/materialize_matched_config.py \\
        --base outputs/C0/resolved_config.json \\
        --override project.name=TumorTrust-VLM-met-holdout-matched-classifier \\
        --override project.output_dir=outputs/met_holdout_matched_classifier \\
        --override data.split_manifest=artifacts/private/cross_evaluations/loso_met_met__trainingdata2_v2_development_split.json \\
        --allow-changed data.split_manifest \\
        --output artifacts/private/overnight/met_holdout_matched_classifier.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tumortrust_vlm.ablation import changed_config_fields
from tumortrust_vlm.config import load_config, with_overrides


def _parse_value(raw: str) -> object:
    """Parse an override's value as YAML so ints/bools/null/strings round-trip correctly."""
    return yaml.safe_load(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Path to the frozen resolved_config.json to match.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="Repeatable. Overrides applied on top of --base before the whitelist check.",
    )
    parser.add_argument(
        "--allow-changed",
        default="",
        help="Comma-separated dotted field names permitted to differ from --base, beyond the "
        "always-ignored project.name/output_dir/operation/trigger/control_scope.",
    )
    parser.add_argument("--output", required=True, help="Where to write the materialized YAML config.")
    args = parser.parse_args()

    base = load_config(args.base)
    overrides: dict[str, object] = {}
    for item in args.override:
        if "=" not in item:
            raise SystemExit(f"--override must be dotted.key=value, got: {item!r}")
        key, raw_value = item.split("=", 1)
        overrides[key] = _parse_value(raw_value)
    candidate = with_overrides(base, overrides)

    allowed = {field.strip() for field in args.allow_changed.split(",") if field.strip()}
    actual_changed = changed_config_fields(base, candidate)
    unexpected = actual_changed - allowed
    if unexpected:
        print(
            "CONFIG_DIFF_ASSERTION_FAILED: unexpected field(s) changed relative to "
            f"{args.base}: {sorted(unexpected)}. Allowed (beyond project.name/output_dir/"
            f"operation/trigger/control_scope): {sorted(allowed) or '(none)'}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    missing = allowed - actual_changed
    if missing:
        print(
            f"CONFIG_DIFF_ASSERTION_FAILED: expected field(s) did not actually change: "
            f"{sorted(missing)}. Check --override values are different from --base.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    print(f"CONFIG_DIFF_ASSERTION_PASSED: changed fields = {sorted(actual_changed)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
