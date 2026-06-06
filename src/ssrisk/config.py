"""Pipeline configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config and environment variables."""
    load_dotenv()
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
