from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tumortrust_vlm.utils import atomic_json_dump


def expected_predictions(mapping_path: Path) -> tuple[set[str], set[str]]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if any(record["split"] == "test" for record in mapping):
        raise ValueError("nnU-Net mapping unexpectedly contains locked test subjects")
    validation = [record for record in mapping if record["split"] == "val"]
    return (
        {f"{record['case_id']}.nii.gz" for record in validation},
        {record["subject_id"] for record in validation},
    )


def prediction_set_exact(path: Path, expected: set[str]) -> bool:
    if not path.is_dir():
        return False
    return {item.name for item in path.glob("*.nii.gz")} == expected


def validation_command(best: bool, python: str = sys.executable) -> list[str]:
    command = [
        python,
        "-m",
        "nnunetv2.run.run_training",
        "501",
        "3d_fullres",
        "0",
        "-p",
        "nnUNetResEncUNetMPlans",
        "--val",
    ]
    if best:
        command.append("--val_best")
    return command


def evaluation_complete(
    cases_path: Path,
    summary_path: Path,
    expected_subjects: set[str],
) -> bool:
    if not cases_path.is_file() or not summary_path.is_file():
        return False
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    observed = {record["subject_id"] for record in cases}
    if len(cases) != len(observed) or observed != expected_subjects:
        raise ValueError(f"nnU-Net evaluation subject mismatch in {cases_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = summary.get("prediction_audit", {})
    if audit.get("prediction_set_exact") is not True:
        raise ValueError(f"nnU-Net evaluation is not exact in {summary_path}")
    if summary.get("locked_test_opened") is not False:
        raise ValueError(f"Locked-test invariant missing in {summary_path}")
    return True


def run_checked(
    command: list[str], env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    completed = subprocess.run(command, env=env, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}: {' '.join(command)}"
        )


def finalize_nnunet(
    root: Path,
    *,
    python: str = sys.executable,
    cuda_visible_devices: str = "1",
) -> dict:
    mapping_path = root / "artifacts/private/nnunet_501_mapping.json"
    expected_files, expected_subjects = expected_predictions(mapping_path)
    fold = (
        root
        / "artifacts/private/nnunet_results/Dataset501_TumorTrust"
        / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0"
    )
    checkpoints = {
        "final": fold / "checkpoint_final.pth",
        "best": fold / "checkpoint_best.pth",
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing nnU-Net checkpoints: {missing}")

    status_path = root / "artifacts/private/nnunet_finalization_status.json"
    status = {
        "schema_version": 1,
        "expected_validation_subjects": len(expected_subjects),
        "locked_test_opened": False,
        "variants": {},
        "state": "running",
    }
    atomic_json_dump(status, status_path)
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(root / "artifacts/private/nnunet_raw"),
            "nnUNet_preprocessed": str(root / "artifacts/private/nnunet_preprocessed"),
            "nnUNet_results": str(root / "artifacts/private/nnunet_results"),
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        }
    )
    live_validation = fold / "validation"
    for variant in ("final", "best"):
        archived = fold / f"validation_{variant}"
        if archived.is_dir() and not prediction_set_exact(archived, expected_files):
            raise ValueError(f"Incomplete archived nnU-Net validation: {archived}")
        if not prediction_set_exact(archived, expected_files):
            if not prediction_set_exact(live_validation, expected_files):
                command = validation_command(
                    best=variant == "best",
                    python=python,
                )
                run_checked(command, env=env, cwd=root)
            if not prediction_set_exact(live_validation, expected_files):
                raise ValueError(
                    f"{variant} validation did not produce the exact frozen prediction set"
                )
            os.replace(live_validation, archived)

        cases_path = root / f"artifacts/private/nnunet_{variant}_val_cases.json"
        official_path = root / f"artifacts/private/nnunet_{variant}_val_official.json"
        summary_path = root / f"artifacts/nnunet_{variant}_val_summary.json"
        if not evaluation_complete(cases_path, summary_path, expected_subjects):
            run_checked(
                [
                    python,
                    "scripts/evaluate_nnunet.py",
                    "--predictions",
                    str(archived),
                    "--case-output",
                    str(cases_path),
                    "--official-output",
                    str(official_path),
                    "--summary-output",
                    str(summary_path),
                ],
                cwd=root,
            )
        if not evaluation_complete(cases_path, summary_path, expected_subjects):
            raise RuntimeError(f"Could not validate nnU-Net {variant} evaluation artifacts")
        status["variants"][variant] = {
            "predictions": str(archived),
            "cases": str(cases_path),
            "summary": str(summary_path),
            "checkpoint": str(checkpoints[variant]),
            "complete": True,
        }
        atomic_json_dump(status, status_path)

    selection_path = root / "artifacts/development_segmentation_baseline_selection.json"
    run_checked(
        [
            python,
            "scripts/freeze_segmentation_baseline.py",
            "--candidate",
            "D3_zscore",
            "outputs/D3_zscore/val_cases.json",
            "outputs/D3_zscore/resolved_config.json",
            "outputs/D3_zscore/checkpoints/best.pt",
            "--candidate",
            "robust_seed2",
            "outputs/segmentation_only_seed2/val_cases.json",
            "outputs/segmentation_only_seed2/resolved_config.json",
            "outputs/segmentation_only_seed2/checkpoints/best.pt",
            "--candidate",
            "nnunet_final",
            str(root / "artifacts/private/nnunet_final_val_cases.json"),
            str(fold / "debug.json"),
            str(checkpoints["final"]),
            "--candidate",
            "nnunet_best",
            str(root / "artifacts/private/nnunet_best_val_cases.json"),
            str(fold / "debug.json"),
            str(checkpoints["best"]),
            "--output",
            str(selection_path),
        ],
        cwd=root,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    status["state"] = "complete"
    status["selection"] = {
        "artifact": str(selection_path),
        "selected_candidate": selection["selected_candidate"],
        "ranking": selection["ranking"],
    }
    atomic_json_dump(status, status_path)
    return status
