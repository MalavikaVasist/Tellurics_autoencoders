"""Shared pytest configuration.

``DataConfig`` resolves its external input paths against ``$TELLURICS_DATA`` and
raises a clear error when the variable is unset. Every test therefore gets an
isolated, empty data root by default, so unit tests never depend on the machine
that runs them. Tests that exercise the missing-variable behaviour delete the
variable themselves via ``monkeypatch.delenv``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path_factory, monkeypatch):
    """Point ``$TELLURICS_DATA`` at a fresh temporary directory per test."""
    root = tmp_path_factory.mktemp("tellurics_data")
    monkeypatch.setenv("TELLURICS_DATA", str(root))
    yield root
