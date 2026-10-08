from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from tumortrust_vlm.config import load_config, with_overrides

_IGNORED_COMPARISON_FIELDS = {
    "project.control_scope",
    "project.name",
    "project.operation",
    "project.output_dir",
    "project.trigger",
}
_MISSING = object()


def _flatten_config(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, object] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_config(child, path))
    return flattened


def changed_config_fields(first: dict, second: dict) -> set[str]:
    left = _flatten_config(first)
    right = _flatten_config(second)
    fields = set(left) | set(right)
    return {
        field
        for field in fields
        if field not in _IGNORED_COMPARISON_FIELDS
        and left.get(field, _MISSING) != right.get(field, _MISSING)
    }


def materialize_ablation_configs(base_path: str, matrix_path: str, output_dir: str) -> list[Path]:
    base = load_config(base_path)
    matrix = load_config(matrix_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    resolved: dict[str, dict] = {}
    for experiment in matrix["experiments"]:
        experiment_id = experiment["id"]
        if experiment_id in resolved:
            raise ValueError(f"Duplicate ablation id: {experiment_id}")
        compare_to = experiment.get("compare_to")
        if compare_to and compare_to not in resolved:
            raise ValueError(
                f"Ablation {experiment_id} compares to unknown or later experiment {compare_to}"
            )
        reference = resolved[compare_to] if compare_to else base
        config = with_overrides(reference, experiment.get("overrides", {}))
        config["project"]["name"] = f"{base['project']['name']}-{experiment_id}"
        config["project"]["output_dir"] = str(
            Path(base["project"]["output_dir"]).parent / experiment_id
        )
        config["project"]["operation"] = experiment.get("operation", "train")
        config["project"]["trigger"] = experiment.get("trigger")
        if compare_to:
            expected = set(experiment.get("expected_changed_fields", []))
            actual = changed_config_fields(config, resolved[compare_to])
            if actual != expected:
                raise ValueError(
                    f"Ablation {experiment_id} violates single-variable declaration versus "
                    f"{compare_to}: expected {sorted(expected)}, found {sorted(actual)}"
                )
        elif experiment.get("expected_changed_fields"):
            raise ValueError(
                f"Ablation {experiment_id} declares changed fields without compare_to"
            )
        resolved[experiment_id] = config
        path = root / f"{experiment_id}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def run_ablation_configs(
    paths: list[Path],
    python: str,
    dry_run: bool = False,
    checkpoint: str | None = None,
    enabled_triggers: set[str] | None = None,
) -> list[dict]:
    results = []
    enabled_triggers = enabled_triggers or set()
    for path in paths:
        config = load_config(path)
        operation = config["project"].get("operation", "train")
        trigger = config["project"].get("trigger")
        if trigger and trigger not in enabled_triggers:
            results.append(
                {
                    "config": str(path),
                    "operation": operation,
                    "command": None,
                    "returncode": None,
                    "status": "skipped_conditional_trigger_not_enabled",
                    "trigger": trigger,
                }
            )
            continue
        if operation == "evaluate":
            if checkpoint is None:
                results.append(
                    {
                        "config": str(path),
                        "operation": operation,
                        "command": None,
                        "returncode": None,
                        "status": "blocked_missing_checkpoint",
                    }
                )
                continue
            command = [
                python,
                "scripts/evaluate.py",
                "--config",
                str(path),
                "--checkpoint",
                checkpoint,
                "--split",
                "val",
            ]
        else:
            command = [python, "scripts/train.py", "--config", str(path)]
        public_command = [Path(command[0]).name, *command[1:]]
        if dry_run:
            results.append(
                {
                    "config": str(path),
                    "operation": operation,
                    "command": public_command,
                    "returncode": None,
                    "status": "dry_run",
                }
            )
            continue
        process = subprocess.run(command, check=False)
        results.append(
            {
                "config": str(path),
                "operation": operation,
                "command": public_command,
                "returncode": process.returncode,
                "status": "completed" if process.returncode == 0 else "failed",
            }
        )
        if process.returncode:
            break
    return results
