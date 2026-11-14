"""YAML config loader and merger."""
from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    out: dict = {}
    for c in configs:
        if c:
            out.update(c)
    return out
