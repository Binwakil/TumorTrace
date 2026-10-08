#!/usr/bin/env python
"""Freeze the one-shot final-test protocol after all development decisions are closed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumortrust_vlm.utils import atomic_json_dump, sha256_file

REQUIRED_FILES = (
    "artifacts/development_segmentation_baseline_selection.json",
    "artifacts/development_segmentation_zscore_three_seed_summary.json",
    "artifacts/shortcut_audit.json",
    "outputs/A1_background_only/val_aggregate.json",
    "artifacts/source_probe_c0.json",
    "artifacts/met_holdout_matched_classifier_test248_summary.json",
    "outputs/M1/val_aggregate.json",
    "outputs/separate_encoder_matched_control_ampoff/val_aggregate.json",
    "artifacts/development_vlm_decoder_comparison.json",
    "artifacts/development_uncertainty_selection.json",
    "artifacts/development_final_evidence_calibration.json",
    "outputs/D3_zscore/resolved_config.json",
    "artifacts/private/ablation_configs_seed2/D3_zscore.yaml",
    "outputs/D3_zscore/checkpoints/best.pt",
    "outputs/C0/resolved_config.json",
    "outputs/C0/checkpoints/best.pt",
    "configs/base.yaml",
    "configs/report_metrics.yaml",
    "configs/uncertainty_referral_closure.yaml",
    "artifacts/private/master_split.json",
    "artifacts/private/master_inventory.json",
    "scripts/run_locked_final_evaluation.sh",
    "scripts/summarize_locked_final.py",
    "scripts/build_final_evidence_calibration.py",
    "scripts/apply_selected_referral.py",
    "scripts/merge_final_classification_evidence.py",
    "scripts/evaluate.py",
    "scripts/extract_core_features.py",
    "scripts/build_report_records.py",
    "scripts/evaluate_deterministic_reports.py",
    "scripts/evaluate_clinical_text_metrics.py",
    "scripts/evaluate_nnunet.py",
    "scripts/export_nnunet_final_test.py",
    "scripts/manage_final_execution.py",
    "environment.yml",
    "environment-report-metrics.yml",
    "environment-report-metrics.lock.txt",
)

NNUNET_CHECKPOINT = (
    "artifacts/private/nnunet_results/Dataset501_TumorTrust/"
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_final.pth"
)
NNUNET_MODEL_FILES = (
    (
        "artifacts/private/nnunet_results/Dataset501_TumorTrust/"
        "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/plans.json"
    ),
    (
        "artifacts/private/nnunet_results/Dataset501_TumorTrust/"
        "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/dataset.json"
    ),
)
EXTERNAL_REPORT_FILES = (
    Path(
        "/data/Projects/llava-tumor-baseline/AutoRG-Brain/"
        "split_meta/train_val_test_split.json"
    ),
    Path(
        "/data/Projects/llava-tumor-baseline/AutoRG-Brain/"
        "report_meta/BraTS_GLI/global_finding.json"
    ),
    Path(
        "/data/Projects/llava-tumor-baseline/AutoRG-Brain/"
        "report_meta/BraTS_MEN/global_finding.json"
    ),
    Path(
        "/data/Projects/llava-tumor-baseline/AutoRG-Brain/"
        "report_meta/BraTS_MET/global_finding.json"
    ),
)


def git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def manifest_key(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/final_protocol_lock.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    receipt = ROOT / "artifacts/private/final_test_execution_receipt.json"
    if receipt.exists():
        raise SystemExit(
            "A final-test execution receipt already exists; protocol cannot be refrozen"
        )
    if output.exists() and not args.force:
        raise SystemExit(f"Protocol lock already exists: {output}")

    uncertainty_path = ROOT / "artifacts/development_uncertainty_selection.json"
    if not uncertainty_path.is_file():
        raise SystemExit("Development uncertainty/referral selection is missing")
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    ensemble_files: tuple[str, ...] = ()
    if (
        uncertainty.get("retain_failure_detection_and_referral")
        and uncertainty.get("selected_candidate", {}).get("method") == "deep_ensemble3"
    ):
        ensemble_files = (
            "outputs/segmentation_zscore_seed0/resolved_config.json",
            "outputs/segmentation_zscore_seed0/checkpoints/best.pt",
            "outputs/segmentation_zscore_seed1/resolved_config.json",
            "outputs/segmentation_zscore_seed1/checkpoints/best.pt",
        )
    package_files = tuple(
        str(path.relative_to(ROOT)) for path in sorted((ROOT / "tumortrust_vlm").rglob("*.py"))
    )
    required = [
        ROOT / value
        for value in (
            *REQUIRED_FILES,
            NNUNET_CHECKPOINT,
            *NNUNET_MODEL_FILES,
            *ensemble_files,
            *package_files,
        )
    ] + list(EXTERNAL_REPORT_FILES)
    missing = [manifest_key(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot freeze protocol; required files are missing: {missing}")

    if not uncertainty.get("final_decision_available"):
        raise SystemExit("Development uncertainty/referral decision is not final")
    referral_retained = bool(uncertainty.get("retain_failure_detection_and_referral"))
    referral = uncertainty.get("development_referral_threshold") if referral_retained else None
    if referral_retained and not referral:
        raise SystemExit("Retained referral has no frozen development threshold")

    files = {
        manifest_key(path): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in required
    }
    payload = {
        "protocol_version": 1,
        "status": "frozen",
        "frozen_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_worktree_note": (
            "The repository contains documented manuscript/experiment changes; immutable final "
            "inputs are identified by hashes below rather than by a clean-tree assertion."
        ),
        "locked_test": {
            "subjects": 320,
            "opened_before_freeze": False,
            "execution_policy": "one_orchestrated_run_only",
            "receipt": "artifacts/private/final_test_execution_receipt.json",
        },
        "retained_framework": {
            "headline_segmentation": "nnU-Net ResEnc final checkpoint",
            "structured_evidence_backbone": "single-task SegResNet D3 z-score",
            "classification": (
                "C0 auxiliary structured output only, source-limited due shortcut-gate failure; "
                "tumor-family prose claim suppressed"
            ),
            "reporting": "deterministic evidence-grounded renderer",
            "learned_vlm": "development ablation only; excluded after grounding failure",
            "referral": {
                "retained": referral_retained,
                "method": uncertainty.get("selected_candidate", {}).get("method"),
                "threshold": referral,
            },
            "radiologist_validation": "optional future enhancement; not an automated-study gate",
        },
        "analysis": {
            "bootstrap_samples": 2000,
            "bootstrap_unit": "patient",
            "confidence_interval": 0.95,
            "paired_tests": "paired patient bootstrap where model outputs share subjects",
            "multiple_comparison": "Holm family-wise correction",
            "random_seed": 20260822,
        },
        "files": files,
    }
    atomic_json_dump(payload, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
