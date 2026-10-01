"""Configuration: bundled defaults, deep-merged with an optional user TOML file."""

from __future__ import annotations

import copy
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

Config = dict[str, Any]


def default_config() -> Config:
    text = resources.files("sangisa").joinpath("default_config.toml").read_text()
    return tomllib.loads(text)


# Tables an override replaces whole instead of merging into: a pad split of
# drums=8,bass=8 means exactly that, not those two on top of the defaults.
REPLACED_TABLES = {"pad_split"}


def merge(base: Config, override: Config) -> Config:
    """Return a copy of ``base`` with ``override`` merged in, recursing into tables."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in REPLACED_TABLES:
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: Config | None = None) -> Config:
    cfg = default_config()
    if path is not None:
        with open(path, "rb") as f:
            cfg = merge(cfg, tomllib.load(f))
    if overrides:
        cfg = merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    split = cfg["kit"]["pad_split"]
    total = sum(split.values())
    if total != cfg["kit"]["pad_count"]:
        raise ValueError(
            f"kit.pad_split adds up to {total} pads but kit.pad_count is {cfg['kit']['pad_count']}"
        )
    if cfg["kit"]["pad_count"] not in (16, 32, 64):
        raise ValueError("kit.pad_count must be 16, 32 or 64")
    if any(n < 0 for n in split.values()):
        raise ValueError("kit.pad_split values must be zero or more")
