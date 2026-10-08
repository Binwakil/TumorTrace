from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tumortrust_vlm.config import config_hash, load_config
from tumortrust_vlm.utils import atomic_json_dump, sha256_file


def _output_dir(config: dict) -> Path:
    return Path(config["project"]["output_dir"])


def _expected_validation_subjects(config: dict) -> set[str]:
    manifest = json.loads(
        Path(config["data"]["split_manifest"]).read_text(encoding="utf-8")
    )
    return {entry["subject_id"] for entry in manifest["entries"] if entry["split"] == "val"}


def training_complete(config_path: Path) -> bool:
    config = load_config(config_path)
    output = _output_dir(config)
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return False
    required = [output / "resolved_config.json", output / "checkpoints" / "best.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Completed training summary has missing artifacts: {missing}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("config_hash") != config_hash(config):
        raise ValueError(f"Training summary config hash mismatch for {config_path}")
    resolved = load_config(output / "resolved_config.json")
    if config_hash(resolved) != config_hash(config):
        raise ValueError(f"Resolved training config mismatch for {config_path}")
    return True


def evaluation_complete(config_path: Path) -> bool:
    config = load_config(config_path)
    output = _output_dir(config)
    aggregate_path = output / "val_aggregate.json"
    cases_path = output / "val_cases.json"
    checkpoint_path = output / "checkpoints" / "best.pt"
    if not aggregate_path.is_file() or not cases_path.is_file():
        return False
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    observed = {case["subject_id"] for case in cases}
    if len(observed) != len(cases) or observed != _expected_validation_subjects(config):
        raise ValueError(f"Validation case set mismatch for {config_path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    provenance = aggregate.get("evaluation_provenance", {})
    if provenance.get("split") != "val" or provenance.get("locked_test_opened") is not False:
        raise ValueError(f"Invalid validation provenance for {config_path}")
    if provenance.get("config_hash") != config_hash(config):
        raise ValueError(f"Evaluation config hash mismatch for {config_path}")
    if provenance.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError(f"Evaluation checkpoint hash mismatch for {config_path}")
    return True


def next_action(config_path: Path) -> tuple[str, Path | None]:
    config = load_config(config_path)
    output = _output_dir(config)
    if not training_complete(config_path):
        last = output / "checkpoints" / "last.pt"
        return ("resume_train", last) if last.is_file() else ("train", None)
    if not evaluation_complete(config_path):
        return "evaluate", output / "checkpoints" / "best.pt"
    return "complete", None


def run_queue(
    registry_path: Path,
    status_path: Path,
    *,
    device: str,
    python: str = sys.executable,
) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    queue_status = {
        "schema_version": 1,
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "device": device,
        "locked_test_opened": False,
        "experiments": [],
    }
    for experiment in registry["experiments"]:
        config_path = Path(experiment["config"])
        while True:
            action, artifact = next_action(config_path)
            state = {
                "id": experiment["id"],
                "config": str(config_path),
                "action": action,
            }
            queue_status["experiments"] = [
                item
                for item in queue_status["experiments"]
                if item["id"] != experiment["id"]
            ] + [state]
            atomic_json_dump(queue_status, status_path)
            if action == "complete":
                break
            if action in {"train", "resume_train"}:
                command = [python, "scripts/train.py", "--config", str(config_path), "--device", device]
                if artifact is not None:
                    command.extend(("--resume", str(artifact)))
            else:
                command = [
                    python,
                    "scripts/evaluate.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(artifact),
                    "--split",
                    "val",
                    "--device",
                    device,
                ]
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                state["action"] = "failed"
                state["failed_command"] = [Path(command[0]).name, *command[1:]]
                state["returncode"] = completed.returncode
                atomic_json_dump(queue_status, status_path)
                raise RuntimeError(
                    f"Stage queue failed for {experiment['id']} with code {completed.returncode}"
                )
    return queue_status
