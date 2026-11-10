"""Resolve the external TelFit distribution (source + LBLRTM runtime data).

TelFit is *not* vendored in this repo: the (heavy) distribution lives outside so
the code repo stays small and pushable, while the TelFit tree can be copied to
scratch on whichever machine runs the simulation.

Location
--------
1. ``$TELFIT_HOME`` if set -- point it at the directory that contains ``src/``,
   ``data/`` and ``telluric_templates/``;
2. otherwise ``<repo>/../Tellurics_data/telfit`` (the default data bundle).

The distribution ships a pre-compiled ``src/FittingUtilities*.so``, so it is
usable without rebuilding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "DEFAULT_TELFIT_HOME",
    "telfit_home",
    "telfit_src",
    "add_telfit_to_path",
]

# src/tellurics/simulation/utils/telfit_home.py -> parents[4] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]

#: Default TelFit distribution directory (sibling of the code repo).
DEFAULT_TELFIT_HOME = _REPO_ROOT.parent / "Tellurics_data" / "telfit"


def telfit_home() -> Path:
    """Return the TelFit distribution directory.

    ``$TELFIT_HOME`` wins; otherwise :data:`DEFAULT_TELFIT_HOME`.
    """
    env = os.environ.get("TELFIT_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_TELFIT_HOME


def telfit_src() -> Path:
    """Return ``<telfit_home>/src`` -- the directory holding the ``telfit`` package."""
    return telfit_home() / "src"


def add_telfit_to_path() -> Path:
    """Prepend ``<telfit_home>/src`` to ``sys.path`` and return it.

    Returns
    -------
    src : Path
        The TelFit ``src`` directory that was added.

    Raises
    ------
    FileNotFoundError
        If ``<telfit_home>/src/telfit`` is missing, so callers get a clear hint
        to set ``$TELFIT_HOME`` instead of a bare ``ImportError``.
    """
    src = telfit_src()
    if not (src / "telfit").is_dir():
        raise FileNotFoundError(
            f"TelFit source not found at {src / 'telfit'}. Set $TELFIT_HOME to "
            "the TelFit distribution directory (the one containing 'src/' and "
            f"'data/'); it currently resolves to {telfit_home()}."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
