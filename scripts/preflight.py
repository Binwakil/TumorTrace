#!/usr/bin/env python
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import monai
import nibabel
import numpy
import scipy
import torch

from tumortrust_vlm.utils import atomic_json_dump


def main() -> None:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": properties.name, "memory_bytes": properties.total_memory})
    revision_process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "monai": monai.__version__,
        "nibabel": nibabel.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "gpus": gpus,
        "git_revision": revision_process.stdout.strip() if revision_process.returncode == 0 else None,
    }
    atomic_json_dump(payload, ROOT / "artifacts" / "environment_snapshot.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
