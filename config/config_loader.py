"""
config/config_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads and validates the YAML configuration file.
Supports CLI overrides via a flat dot-notation string list, e.g.
    training.lr=0.01  model.backbone=mobilenetv2

Supports hardware profiles via:
    load_config(profile="apple_silicon")
which deep-merges config/profiles/{profile}.yaml AFTER base config
but BEFORE CLI overrides (so CLI always wins).
"""
from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default config path (relative to this file's directory)
_DEFAULT_CONFIG  = Path(__file__).parent / "config.yaml"
_PROFILES_DIR    = Path(__file__).parent / "profiles"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, keys: List[str], value: Any) -> None:
    """Set d[keys[0]][keys[1]]... = value, creating sub-dicts as needed."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Auto-cast obvious types
    raw = value
    if isinstance(raw, str):
        if raw.lower() == "true":
            raw = True
        elif raw.lower() == "false":
            raw = False
        elif raw.lower() == "null":
            raw = None
        else:
            try:
                raw = int(raw)
            except ValueError:
                try:
                    raw = float(raw)
                except ValueError:
                    pass
    d[keys[-1]] = raw


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load YAML config, optionally applying a hardware profile and CLI overrides.

    Merge order (later wins):
        base config  <  profile  <  CLI overrides

    Parameters
    ----------
    config_path : str or None
        Path to a YAML file.  Defaults to config/config.yaml.
    overrides : list[str] or None
        Key=value pairs like ["training.lr=0.01", "model.backbone=mobilenetv2"].
    profile : str or None
        Hardware profile name.  Loads config/profiles/{profile}.yaml and
        deep-merges it over the base config before applying CLI overrides.
        Choices: "apple_silicon" | "cuda_gpu" | "cpu_debug"

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply hardware profile (base → profile → CLI overrides)
    if profile is not None:
        profile_path = _PROFILES_DIR / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Hardware profile '{profile}' not found: {profile_path}"
            )
        with open(profile_path, "r") as f:
            profile_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, profile_cfg)

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item!r}")
            dotkey, _, val = item.partition("=")
            keys = dotkey.strip().split(".")
            _set_nested(cfg, keys, val)

    return cfg
