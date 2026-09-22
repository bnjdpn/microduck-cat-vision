from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
MODEL_ROOT = ARTIFACT_ROOT / "model"
SIMULATOR_DIST = PROJECT_ROOT / "simulator" / "app" / "dist"


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

