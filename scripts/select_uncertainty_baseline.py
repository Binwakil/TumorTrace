#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen development-only uncertainty/referral selection protocol."
    )
    parser.add_argument(
        "--config", default="configs/uncertainty_referral_closure.yaml"
    )
    parser.add_argument("--shift-summary")
    parser.add_argument(
        "--output", default="artifacts/development_uncertainty_selection.json"
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidates = []
    for specification in config["candidate_signals"]:
        artifact_path = ROOT / specification["artifact"]
        summary = load_json(artifact_path)
        method = specification["method"]
        if summary["method"] != method:
            raise ValueError(
                f"Method mismatch for {artifact_path}: {summary['method']} != {method}"
            )
        signal = specification["signal"]
        signal_summary = summary["overall"][signal]
        referral = signal_summary["referral_at_80pct"]
        candidate = {
            "method": method,
            "signal": signal,
            "artifact": specification["artifact"],
            "artifact_sha256": sha256_file(artifact_path),
            "failure_auroc": float(signal_summary["failure_auroc"]),
            "failure_auprc": float(signal_summary["failure_auprc"]),
            "relative_risk_reduction_at_80pct": float(
                referral["selective_relative_risk_reduction"]
            ),
            "random_referral_p_value": float(referral["random_referral_p_value"]),
        }
        if not all(
            np.isfinite(candidate[field])
            for field in (
                "failure_auroc",
                "failure_auprc",
                "relative_risk_reduction_at_80pct",
                "random_referral_p_value",
            )
        ):
            raise ValueError(f"Non-finite selection metric for {method}")
        candidates.append(candidate)

    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["failure_auroc"],
            -row["relative_risk_reduction_at_80pct"],
            row["method"],
        ),
    )
    selected = ranked[0]
    feature_path = ROOT / "artifacts/private/trust" / f"{selected['method']}_features.json"
    feature_rows = json.loads(feature_path.read_text(encoding="utf-8"))
    signal_field = {
        "predictive_entropy": "segmentation_predictive_entropy_p95",
        "mutual_information": "segmentation_mutual_information_p95",
    }[selected["signal"]]
    signal_values = np.sort(
        np.asarray(
            [row["uncertainty"][signal_field] for row in feature_rows], dtype=float
        )
    )
    retained_count = max(1, int(np.ceil(len(signal_values) * 0.80)))
    referral_threshold = float(signal_values[retained_count - 1])
    gates = config["retention_gates"]
    iid_gates = {
        "failure_auroc": (
            selected["failure_auroc"]
            >= gates["iid_failure_detection_auroc_minimum"]
        ),
        "risk_reduction": (
            selected["relative_risk_reduction_at_80pct"]
            >= gates["relative_risk_reduction_at_80pct_coverage_minimum"]
        ),
        "beats_random_referral": (
            selected["random_referral_p_value"]
            <= gates["random_referral_p_value_maximum"]
        ),
    }

    shift_result = None
    shift_gates: dict[str, bool] = {}
    if args.shift_summary:
        shift_path = Path(args.shift_summary)
        shift_result = load_json(shift_path)
        if shift_result["method"] != selected["method"]:
            raise ValueError("Shift summary does not describe the selected method")
        for shift_name in config["preregistered_shifts"]:
            if shift_name not in shift_result["shift_detection"]:
                raise ValueError(f"Missing preregistered shift: {shift_name}")
            auroc = shift_result["shift_detection"][shift_name][
                f"{selected['signal']}_auroc"
            ]
            shift_gates[shift_name] = (
                float(auroc) >= gates["shift_detection_auroc_minimum_each"]
            )
        shift_result = {
            "artifact": str(shift_path),
            "artifact_sha256": sha256_file(shift_path),
            "detection": shift_result["shift_detection"],
        }

    final_decision_available = args.shift_summary is not None
    retained = (
        all(iid_gates.values()) and all(shift_gates.values())
        if final_decision_available
        else None
    )
    result = {
        "protocol": args.config,
        "protocol_sha256": sha256_file(config_path),
        "scope": "development_only",
        "locked_final_test_opened": False,
        "ranked_candidates": ranked,
        "selected_candidate": selected,
        "development_referral_threshold": {
            "signal": selected["signal"],
            "uncertainty_field": signal_field,
            "uncertainty_threshold": referral_threshold,
            "coverage": 0.80,
            "comparison": "refer_if_strictly_greater_than_threshold",
            "features": str(feature_path.relative_to(ROOT)),
            "features_sha256": sha256_file(feature_path),
        },
        "iid_gates": iid_gates,
        "shift_result": shift_result,
        "shift_gates": shift_gates,
        "final_decision_available": final_decision_available,
        "retain_failure_detection_and_referral": retained,
        "referral_threshold_frozen": bool(retained),
        "failure_action": config["failure_action"] if retained is False else None,
    }
    atomic_json_dump(result, ROOT / args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
