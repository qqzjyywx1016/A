"""YAML configuration loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root for this package layout."""

    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and attach the resolved project root."""

    root = project_root()
    path = Path(config_path) if config_path else root / "config" / "default.yaml"
    if not path.is_absolute():
        path = root / path
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    config = deepcopy(config)
    config["_project_root"] = str(root)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to project root unless already absolute."""

    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(config.get("_project_root", project_root()))
    return root / path
