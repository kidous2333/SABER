"""
Config loader utilities — merge YAML configs + apply GUI overrides.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_merged_config(*paths: str) -> dict:
    """
    Load and merge multiple YAML files in order.
    Later files' top-level keys override earlier ones.
    """
    merged: Dict[str, Any] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged.update(data)
    return merged


def save_config(config: dict, path: str) -> None:
    """Save config dict to YAML file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_config_file(cfg: dict, key: str, default: str = "") -> str:
    """Resolve a config file path, supporting both absolute and relative paths."""
    val = cfg.get(key, default)
    if not val:
        return default
    return val


def resolve_path(base_dir: Path, path_str: str) -> Path:
    """Resolve a path relative to base_dir if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p


def list_config_files(config_dir: str = "config") -> list:
    """List all YAML files in a config directory."""
    p = Path(config_dir)
    if not p.is_dir():
        return []
    return sorted([str(f) for f in p.glob("*.yaml")] + [str(f) for f in p.glob("*.yml")])
