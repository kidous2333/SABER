"""
GUI Settings Manager — separate config storage independent of project YAML files.

Stores all GUI tab settings in a single JSON file:
    gui_settings.json

Flow:
  1. First launch: load defaults from tab schemas → save to gui_settings.json
  2. Subsequent launches: load from gui_settings.json directly
  3. Configure dialog edits → save to gui_settings.json
  4. Project YAML files are NEVER modified
"""

import json
from pathlib import Path
from typing import Dict, Optional

# Use absolute path anchored to project root to survive os.chdir() calls
_project_root = Path(__file__).resolve().parent.parent.parent
SETTINGS_PATH = _project_root / "gui_settings.json"


def load_settings() -> dict:
    """Load all GUI settings. Returns empty dict if file doesn't exist."""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(settings: dict):
    """Save all GUI settings to disk."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def get_tab_settings(tab_key: str) -> dict:
    """Get settings for a specific tab."""
    all_settings = load_settings()
    return all_settings.get(tab_key, {})


def save_tab_settings(tab_key: str, values: dict):
    """Save settings for a specific tab."""
    all_settings = load_settings()
    all_settings[tab_key] = values
    save_settings(all_settings)
