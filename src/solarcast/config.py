"""Load the YAML configuration."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
