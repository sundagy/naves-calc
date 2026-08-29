from __future__ import annotations

from pathlib import Path

import yaml

from src.models import CanopyInput


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_presets() -> dict[str, dict]:
    return load_yaml(CONFIG_DIR / "presets.yaml")


def load_sections() -> list[str]:
    data = load_yaml(CONFIG_DIR / "sections.yaml")
    return list(data["sections"])


def load_normative() -> dict:
    return load_yaml(CONFIG_DIR / "normative.yaml")


def input_from_preset(name: str) -> CanopyInput:
    presets = load_presets()
    if name not in presets:
        raise KeyError(f"Unknown preset: {name}")
    return CanopyInput(preset_name=name, **presets[name])

