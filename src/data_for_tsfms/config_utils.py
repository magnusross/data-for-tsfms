from __future__ import annotations
from pathlib import Path
import yaml
from typer_config import conf_callback_factory

_last_loaded_config: dict = {}


def _load_with_bases(param_value: str) -> dict:
    """Loader for typer-config: loads YAML, merging any _base files first."""
    global _last_loaded_config

    def _merge(path: Path) -> dict:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        base_paths = data.pop("_base", [])
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        merged: dict = {}
        for base in base_paths:
            merged.update(_merge(Path(base)))
        merged.update(data)
        return merged

    result = _merge(Path(param_value))
    _last_loaded_config = result
    return result


def get_loaded_config() -> dict:
    """Return the full config dict loaded by the last yaml_conf_callback invocation."""
    return dict(_last_loaded_config)


yaml_conf_callback = conf_callback_factory(_load_with_bases)
