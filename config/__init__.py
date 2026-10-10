"""
config/__init__.py
Exposes load_config so callers can do:  from config import load_config
"""
from .config_loader import load_config

__all__ = ["load_config"]
