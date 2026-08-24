"""Small shared helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(name: str) -> Optional[torch.dtype]:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[name]


class JsonlLogger:
    """Append-only JSONL metric log."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


def human_count(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return str(n)


__all__ = ["JsonlLogger", "autocast_dtype", "human_count", "resolve_device", "set_seed"]
