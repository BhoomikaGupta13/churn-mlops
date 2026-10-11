"""Loads params.yaml so every script reads settings from one place."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: Path | str = ROOT / "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
