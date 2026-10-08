from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from tumortrust_vlm.config import load_config

EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)\s*$")
FLOAT_PATTERNS = {
    "learning_rate": re.compile(r"Current learning rate:\s+([-+\d.eE]+)"),
    "train_loss": re.compile(r"train_loss\s+([-+\d.eE]+)"),
    "val_loss": re.compile(r"val_loss\s+([-+\d.eE]+)"),
    "best_ema_pseudo_dice": re.compile(r"New best EMA pseudo Dice:\s+([-+\d.eE]+)"),
    "epoch_seconds": re.compile(r"Epoch time:\s+([-+\d.eE]+)\s+s"),
}
PSEUDO_DICE_PATTERN = re.compile(r"Pseudo dice\s+\[([^]]+)]")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def d3_status(history_path: Path, config_path: Path) -> dict[str, Any]:
    history = read_json(history_path, [])
    config = read_json(config_path, {})
    validations = [record for record in history if record.get("val_macro_dice") is not None]
    best = max(validations, key=lambda record: record["val_macro_dice"], default=None)
    latest_validation = validations[-1] if validations else None
    training = config.get("training", {})
    return {
        "epoch": int(history[-1]["epoch"]) if history else 0,
        "epochs": int(training.get("epochs", 0)),
        "loss": history[-1].get("loss") if history else None,
        "latest_validation_epoch": latest_validation.get("epoch") if latest_validation else None,
        "latest_macro_dice": (
            latest_validation.get("val_macro_dice") if latest_validation else None
        ),
        "best_validation_epoch": best.get("epoch") if best else None,
        "best_macro_dice": best.get("val_macro_dice") if best else None,
        "validate_every": int(training.get("validate_every", 0)),
        "patience": int(training.get("patience", 0)),
    }


def nnunet_status(log_paths: list[Path], epochs: int = 1000) -> dict[str, Any]:
    logs = []
    for path in log_paths:
        try:
            logs.append((path.stat().st_mtime, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    if not logs:
        return {"epoch": 0, "epochs": epochs}
    logs.sort(key=lambda item: item[0])
    latest_text = logs[-1][1]
    all_text = "\n".join(text for _, text in logs)
    status: dict[str, Any] = {"epoch": 0, "epochs": epochs}
    epoch_times: list[float] = []
    for line in latest_text.splitlines():
        if match := EPOCH_PATTERN.search(line.strip()):
            status["epoch"] = int(match.group(1))
        if match := PSEUDO_DICE_PATTERN.search(line):
            status["pseudo_dice"] = [float(value.strip()) for value in match.group(1).split(",")]
        for key, pattern in FLOAT_PATTERNS.items():
            if match := pattern.search(line):
                value = float(match.group(1))
                if key == "epoch_seconds":
                    epoch_times.append(value)
                else:
                    status[key] = value
    best_values = [
        float(match.group(1))
        for match in FLOAT_PATTERNS["best_ema_pseudo_dice"].finditer(all_text)
    ]
    if best_values:
        status["best_ema_pseudo_dice"] = max(best_values)
    if epoch_times:
        status["epoch_seconds"] = median(epoch_times[-20:])
    return status


def stage5_status(root: Path, registry_path: Path, queue_status_path: Path) -> dict[str, Any]:
    registry = read_json(registry_path, {"experiments": []})
    queue = read_json(queue_status_path, {"experiments": []})
    queued = {record["id"]: record for record in queue.get("experiments", [])}
    experiments = []
    for record in registry.get("experiments", []):
        config_path = Path(record["config"])
        if not config_path.is_absolute():
            config_path = root / config_path
        try:
            config = load_config(config_path)
        except (FileNotFoundError, OSError, TypeError):
            continue
        output = Path(config["project"]["output_dir"])
        if not output.is_absolute():
            output = root / output
        history = read_json(output / "history.json", [])
        summary = read_json(output / "summary.json", {})
        aggregate_complete = (output / "val_aggregate.json").is_file()
        mode = config["model"].get("joint_weighting", "fixed")
        metric = "balanced_accuracy" if mode == "classification_only" else "macro_dice"
        validations = [
            item for item in history if item.get(f"val_{metric}") is not None
        ]
        history_best = (
            max(item[f"val_{metric}"] for item in validations)
            if validations
            else None
        )
        queue_record = queued.get(record["id"], {})
        experiments.append(
            {
                "id": record["id"],
                "mode": mode,
                "action": queue_record.get("action", "gated_on_M3"),
                "epoch": int(history[-1]["epoch"]) if history else 0,
                "epochs": int(config["training"].get("epochs", 0)),
                "selection_metric": metric,
                "latest_validation": (
                    validations[-1].get(f"val_{metric}") if validations else None
                ),
                "best_validation": summary.get(
                    f"best_validation_{metric}", history_best
                ),
                "training_complete": bool(summary),
                "evaluation_complete": aggregate_complete,
                "complete": bool(summary) and aggregate_complete,
            }
        )
    current = next((item for item in experiments if not item["complete"]), None)
    if current is None and experiments:
        current = experiments[-1]
    return {
        "completed_experiments": sum(item["complete"] for item in experiments),
        "total_experiments": len(experiments),
        "current": current,
        "experiments": experiments,
    }


def matching_processes(needles: list[str]) -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,etimes=,stat=,command="], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    records = []
    for line in output.splitlines():
        if all(needle in line for needle in needles) and "monitor.py" not in line:
            parts = line.strip().split(maxsplit=3)
            if len(parts) == 4:
                records.append(
                    {
                        "pid": int(parts[0]),
                        "elapsed_seconds": int(parts[1]),
                        "process_state": parts[2],
                        "command": parts[3],
                    }
                )
    return records


def gpu_status() -> list[dict[str, Any]]:
    fields = ["index", "name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu"]
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    result = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        result.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "memory_used_mib": int(values[2]),
                "memory_total_mib": int(values[3]),
                "utilization_percent": int(values[4]),
                "temperature_c": int(values[5]),
            }
        )
    return result


def progress_bar(current: int, total: int, width: int = 28) -> str:
    fraction = min(max(current / total, 0.0), 1.0) if total else 0.0
    filled = round(width * fraction)
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction * 100:5.1f}%"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "learning..."
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_optional(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


@dataclass
class ProgressRates:
    samples: dict[str, tuple[int, float]] = field(default_factory=dict)
    rates: dict[str, deque[float]] = field(default_factory=dict)

    def eta(self, name: str, current: int, total: int, now: float) -> float | None:
        previous = self.samples.get(name)
        if previous and current > previous[0]:
            seconds_per_unit = (now - previous[1]) / (current - previous[0])
            self.rates.setdefault(name, deque(maxlen=8)).append(seconds_per_unit)
        if previous is None or current != previous[0]:
            self.samples[name] = (current, now)
        values = self.rates.get(name)
        return median(values) * max(total - current, 0) if values else None


def collect_status(root: Path, rates: ProgressRates) -> dict[str, Any]:
    now = time.time()
    d3 = d3_status(root / "outputs/D3_zscore/history.json", root / "outputs/D3_zscore/resolved_config.json")
    nn_root = (
        root
        / "artifacts/private/nnunet_results/Dataset501_TumorTrust"
        / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0"
    )
    nnunet = nnunet_status(sorted(nn_root.glob("training_log*.txt")))
    nn_mapping = read_json(root / "artifacts/private/nnunet_501_mapping.json", [])
    nnunet["validation_total"] = sum(
        record.get("split") == "val" for record in nn_mapping
    )
    nnunet["training_complete"] = (nn_root / "checkpoint_final.pth").is_file()
    nnunet["validation_variant"] = (
        "best" if (nn_root / "validation_final").is_dir() else "final"
    )
    nnunet["validation_predictions"] = len(
        list((nn_root / "validation").glob("*.nii.gz"))
    )
    stage5 = stage5_status(
        root,
        root / "artifacts/private/stage5_configs/registry.json",
        root / "artifacts/private/stage5_queue_status.json",
    )
    d3_processes = matching_processes(["scripts/train.py", "D3_zscore"])
    d3_evaluators = matching_processes(["scripts/evaluate.py", "D3_zscore"])
    nn_processes_by_pid = {
        record["pid"]: record
        for record in [
            *matching_processes(["nnUNetv2_train", "501", "3d_fullres"]),
            *matching_processes(
                ["nnunetv2.run.run_training", "501", "3d_fullres"]
            ),
        ]
    }
    nn_processes = list(nn_processes_by_pid.values())
    nn_evaluators = matching_processes(["scripts/evaluate_nnunet.py"])
    d3["active"] = bool(d3_processes)
    d3["evaluation_active"] = bool(d3_evaluators)
    d3["pid"] = min(
        (record["pid"] for record in [*d3_processes, *d3_evaluators]), default=None
    )
    d3["workers"] = max(len(d3_processes) - 1, 0)
    d3["eta_seconds"] = rates.eta("d3", d3["epoch"], d3["epochs"], now)
    nnunet["active"] = bool(nn_processes)
    nnunet["evaluation_active"] = bool(nn_evaluators)
    nnunet["pid"] = min(
        (record["pid"] for record in [*nn_processes, *nn_evaluators]), default=None
    )
    nnunet["workers"] = max(len(nn_processes) - 1, 0)
    if nnunet["active"] and nnunet["training_complete"]:
        nnunet["phase"] = f"{nnunet['validation_variant']}_validation"
        nnunet["eta_seconds"] = rates.eta(
            f"nnunet_validation_{nnunet['validation_variant']}",
            nnunet["validation_predictions"],
            nnunet["validation_total"],
            now,
        )
    elif nnunet["active"]:
        nnunet["phase"] = "training"
    elif nnunet["evaluation_active"]:
        evaluator_commands = " ".join(record["command"] for record in nn_evaluators)
        nnunet["phase"] = (
            "best_metrics" if "validation_best" in evaluator_commands else "final_metrics"
        )
        nnunet["eta_seconds"] = None
    else:
        nnunet["phase"] = "stopped"
    if nnunet["phase"] == "training" and nnunet.get("epoch_seconds") is not None:
        nnunet["eta_seconds"] = nnunet["epoch_seconds"] * max(
            nnunet["epochs"] - nnunet["epoch"], 0
        )
    elif nnunet["phase"] == "training":
        nnunet["eta_seconds"] = rates.eta(
            "nnunet", nnunet["epoch"], nnunet["epochs"], now
        )
    stage5_processes = matching_processes(["scripts/run_stage5_queue.py"])
    stage5["active"] = bool(stage5_processes)
    stage5["pid"] = min((record["pid"] for record in stage5_processes), default=None)
    if stage5["current"]:
        stage5["current"]["eta_seconds"] = rates.eta(
            f"stage5_{stage5['current']['id']}",
            stage5["current"]["epoch"],
            stage5["current"]["epochs"],
            now,
        )
    disk = shutil.disk_usage(root)
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "d3_zscore": d3,
        "nnunet": nnunet,
        "stage5": stage5,
        "gpus": gpu_status(),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "used_percent": disk.used / disk.total * 100,
        },
    }


def render_status(status: dict[str, Any], interval: float) -> str:
    d3 = status["d3_zscore"]
    nnunet = status["nnunet"]
    stage5 = status["stage5"]
    if d3["active"]:
        d3_state = "TRAINING"
    elif d3["evaluation_active"]:
        d3_state = "FULL VALIDATION"
    elif d3["epoch"] >= d3["epochs"]:
        d3_state = "TRAINING COMPLETE"
    else:
        d3_state = "STOPPED"
    nn_states = {
        "training": "TRAINING",
        "final_validation": "FINAL VALIDATION",
        "best_validation": "BEST VALIDATION",
        "final_metrics": "FINAL METRIC EVALUATION",
        "best_metrics": "BEST METRIC EVALUATION",
        "stopped": "STOPPED",
    }
    nn_state = nn_states[nnunet["phase"]]
    lines = [
        f"TumorTrust-VLM live monitor  |  {status['updated_at']}",
        "=" * 78,
        "",
        f"D3 z-score SegResNet  {d3_state}  pid={d3['pid'] or '-'}  workers={d3['workers']}",
        f"  {progress_bar(d3['epoch'], d3['epochs'])}  epoch {d3['epoch']}/{d3['epochs']}",
        (
            f"  train loss {format_optional(d3.get('loss'))}  |  "
            f"latest val {format_optional(d3.get('latest_macro_dice'))}"
            f" @ {d3.get('latest_validation_epoch') or '-'}  |  "
            f"best {format_optional(d3.get('best_macro_dice'))}"
            f" @ {d3.get('best_validation_epoch') or '-'}"
        ),
        f"  ETA to maximum budget: {format_duration(d3.get('eta_seconds'))}",
        "",
        (
            f"Official nnU-Net ResEnc-M  {nn_state}  pid={nnunet['pid'] or '-'}  "
            f"workers={nnunet['workers']}"
        ),
        *(
            [
                (
                    f"  {progress_bar(nnunet['validation_predictions'], nnunet['validation_total'])}  "
                    f"cases {nnunet['validation_predictions']}/{nnunet['validation_total']}"
                )
            ]
            if nnunet["phase"] in {"final_validation", "best_validation"}
            else ["  248/248 masks archived; computing project and official metrics"]
            if nnunet["phase"] in {"final_metrics", "best_metrics"}
            else [
                (
                    f"  {progress_bar(nnunet['epoch'], nnunet['epochs'])}  "
                    f"epoch {nnunet['epoch']}/{nnunet['epochs']}"
                )
            ]
        ),
        (
            f"  pseudo Dice {nnunet.get('pseudo_dice', 'n/a')}  |  "
            f"best EMA {format_optional(nnunet.get('best_ema_pseudo_dice'))}  |  "
            f"LR {format_optional(nnunet.get('learning_rate'), 6)}"
        ),
        (
            f"  median epoch {format_duration(nnunet.get('epoch_seconds'))}  |  "
            f"ETA {format_duration(nnunet.get('eta_seconds'))}"
        ),
        "",
        (
            f"Stage 5 classification/multitask queue  "
            f"{'ACTIVE' if stage5['active'] else 'WAITING'}  pid={stage5['pid'] or '-'}"
        ),
        (
            f"  experiments {stage5['completed_experiments']}/{stage5['total_experiments']} complete"
        ),
    ]
    current = stage5.get("current")
    if current:
        lines.extend(
            [
                (
                    f"  current {current['id']} ({current['mode']})  "
                    f"action={current['action']}"
                ),
                (
                    f"  {progress_bar(current['epoch'], current['epochs'])}  "
                    f"epoch {current['epoch']}/{current['epochs']}  |  "
                    f"latest {current['selection_metric']} "
                    f"{format_optional(current.get('latest_validation'))}  |  "
                    f"best {format_optional(current.get('best_validation'))}"
                ),
                f"  ETA to maximum budget: {format_duration(current.get('eta_seconds'))}",
            ]
        )
    lines.extend(
        [
            "",
            "GPUs",
        ]
    )
    for gpu in status["gpus"]:
        lines.append(
            f"  GPU {gpu['index']}: {gpu['utilization_percent']:3d}%  "
            f"{gpu['memory_used_mib']:5d}/{gpu['memory_total_mib']} MiB  "
            f"{gpu['temperature_c']} C  {gpu['name']}"
        )
    disk = status["disk"]
    lines.extend(
        [
            "",
            (
                f"Disk: {disk['used_percent']:.1f}% used  |  "
                f"{disk['free_bytes'] / (1024**3):.1f} GiB free"
            ),
            f"Refresh: {interval:g}s  |  Ctrl-C exits only the monitor; training continues.",
        ]
    )
    return "\n".join(lines)
