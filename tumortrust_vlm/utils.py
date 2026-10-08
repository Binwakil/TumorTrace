from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Capture process-level random streams for an exact between-epoch resume."""
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a checkpoint random state while accepting older checkpoints."""
    if not state:
        return
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu")
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [
            torch.as_tensor(cuda_state, dtype=torch.uint8, device="cpu")
            for cuda_state in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite values to strict-JSON representations."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading large checkpoints into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact_provenance(root: str | Path) -> dict[str, str]:
    """Hash configuration, tokenizer, and weight files for a local language model."""
    model_root = Path(root)
    files = sorted(
        path
        for pattern in ("*.json", "*.model", "*.safetensors", "*.bin")
        for path in model_root.glob(pattern)
        if path.is_file()
    )
    if not files:
        raise ValueError(f"No base-model provenance files found under {model_root}")
    return {str(path.relative_to(model_root)): sha256_file(path) for path in files}


def git_revision(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
