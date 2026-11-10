
"""Resolve external data paths using TELLURICS_DATA."""

import os
from pathlib import Path

__all__ = ["DATA_ROOT_ENV", "data_root", "resolve_data_path", "expand_env_vars"]

#: Environment variable naming the external data root.
DATA_ROOT_ENV = "TELLURICS_DATA"


def expand_env_vars(value):
    """Expand environment variables in strings, dictionaries, and lists."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value


def data_root():
    """Return the configured external data directory."""
    root = os.environ.get(DATA_ROOT_ENV)
    if not root:
        raise RuntimeError(
            f"Please set the {DATA_ROOT_ENV} environment variable, e.g.\n"
            f'    export {DATA_ROOT_ENV}="$HOME/Documents/Tellurics_data"'
        )

    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")

    return path


def resolve_data_path(path):
    """Resolve a data path relative to TELLURICS_DATA."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else data_root() / path
