"""Tests for machine-independent data paths (``$TELLURICS_DATA``).

Acceptance: the same commit, code and tracked YAML must work on any machine;
only ``TELLURICS_DATA`` and the external data files differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tellurics.config.data import DataConfig
from tellurics.config.experiment import load_config
from tellurics.config.paths import (
    DATA_ROOT_ENV,
    data_root,
    expand_env_vars,
    resolve_data_path,
)
from tests._bounds import BOUNDS

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
# Every tracked manifest, relative to experiments/ (multiple_frames/, single_frame/).
TRACKED_MANIFESTS = sorted(
    str(p.relative_to(EXPERIMENTS)) for p in EXPERIMENTS.glob("*/*.yaml")
)


# --------------------------------------------------------------------------- #
# data_root()
# --------------------------------------------------------------------------- #
def test_data_root_reads_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert data_root() == tmp_path.resolve()


def test_data_root_expands_user(monkeypatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, "~")
    assert data_root() == Path.home().resolve()


def test_data_root_missing_raises_actionable_error(monkeypatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=DATA_ROOT_ENV):
        data_root()


def test_data_root_nonexistent_directory_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        data_root()


# --------------------------------------------------------------------------- #
# resolve_data_path()
# --------------------------------------------------------------------------- #
def test_relative_path_resolves_against_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert resolve_data_path("phoenix/convolved") == tmp_path.resolve() / "phoenix/convolved"


def test_absolute_path_is_untouched(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    absolute = Path("/scratch/someone/Tellurics_data/phoenix/convolved")
    assert resolve_data_path(absolute) == absolute


def test_absolute_path_needs_no_env_var(monkeypatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    absolute = Path("/scratch/someone/Tellurics_data/x.h5")
    assert resolve_data_path(absolute) == absolute


# --------------------------------------------------------------------------- #
# DataConfig wiring
# --------------------------------------------------------------------------- #
def test_data_config_resolves_relative_inputs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = DataConfig(
        param_bounds=BOUNDS,
        night_h5="telluric_timeseries/a.h5",
        phoenix_dir="phoenix/convolved",
    )
    assert cfg.night_h5 == tmp_path.resolve() / "telluric_timeseries/a.h5"
    assert cfg.phoenix_dir == tmp_path.resolve() / "phoenix/convolved"


def test_data_config_defaults_are_relative_on_disk(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = DataConfig(param_bounds=BOUNDS)
    assert cfg.night_h5.is_absolute() and cfg.phoenix_dir.is_absolute()
    assert cfg.night_h5.is_relative_to(tmp_path.resolve())
    assert cfg.phoenix_dir.is_relative_to(tmp_path.resolve())


def test_data_config_requires_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=DATA_ROOT_ENV):
        DataConfig(param_bounds=BOUNDS)


# --------------------------------------------------------------------------- #
# End-to-end: one tracked manifest, two machines
# --------------------------------------------------------------------------- #
def test_tracked_manifest_resolves_per_machine(monkeypatch, tmp_path) -> None:
    manifest = EXPERIMENTS / "multiple_frames" / "experiment1_smoke.yaml"

    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "machineA"))
    (tmp_path / "machineA").mkdir()
    a = load_config(manifest).data

    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "machineB"))
    (tmp_path / "machineB").mkdir()
    b = load_config(manifest).data

    assert a.root == (tmp_path / "machineA").resolve()
    assert b.root == (tmp_path / "machineB").resolve()

    expected = Path("telluric_timeseries/telluric_templates_nights.h5")
    assert a.night_h5 == (tmp_path / "machineA").resolve() / expected
    assert b.night_h5 == (tmp_path / "machineB").resolve() / expected
    assert a.phoenix_dir == (tmp_path / "machineA").resolve() / "phoenix/convolved"
    assert b.phoenix_dir == (tmp_path / "machineB").resolve() / "phoenix/convolved"


@pytest.mark.parametrize("name", TRACKED_MANIFESTS)
def test_tracked_manifests_have_no_absolute_data_paths(name: str) -> None:
    """Tracked YAML must stay machine-independent."""
    raw = yaml.safe_load((EXPERIMENTS / name).read_text()) or {}
    data = raw.get("data", {})
    for key in ("night_h5", "phoenix_dir"):
        assert key in data, f"{name}: data.{key} is missing"
        assert not Path(data[key]).is_absolute(), (
            f"{name}: data.{key} must be relative to $TELLURICS_DATA "
            f"(got {data[key]!r})"
        )


@pytest.mark.parametrize("name", TRACKED_MANIFESTS)
def test_tracked_manifests_declare_the_env_root(name: str) -> None:
    """``data.root`` is the ${TELLURICS_DATA} placeholder, not a machine path."""
    raw = yaml.safe_load((EXPERIMENTS / name).read_text()) or {}
    assert raw["data"]["root"] == "${TELLURICS_DATA}"


# --------------------------------------------------------------------------- #
# ${VAR} expansion (done by load_config; YAML does not expand variables)
# --------------------------------------------------------------------------- #
def test_expand_env_vars_scalar(monkeypatch) -> None:
    monkeypatch.setenv("MY_ROOT", "/scratch/me/data")
    assert expand_env_vars("${MY_ROOT}/phoenix") == "/scratch/me/data/phoenix"
    assert expand_env_vars("$MY_ROOT/phoenix") == "/scratch/me/data/phoenix"


def test_expand_env_vars_is_recursive(monkeypatch) -> None:
    monkeypatch.setenv("A", "x")
    payload = {"root": "${A}", "paths": ["${A}/y", {"z": "pre-${A}"}], "n": 3}
    assert expand_env_vars(payload) == {
        "root": "x",
        "paths": ["x/y", {"z": "pre-x"}],
        "n": 3,
    }


def test_expand_env_vars_unset_is_left_literal(monkeypatch) -> None:
    """``os.path.expandvars`` leaves unknown vars as-is (handled downstream)."""
    monkeypatch.delenv("NOPE", raising=False)
    assert expand_env_vars("${NOPE}/x") == "${NOPE}/x"


def test_load_config_expands_root_and_ignores_cwd(monkeypatch, tmp_path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(root))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    cfg = load_config(EXPERIMENTS / "multiple_frames" / "experiment1_smoke.yaml").data
    assert cfg.root == root.resolve()
    assert cfg.night_h5 == root.resolve() / "telluric_timeseries/telluric_templates_nights.h5"
    assert cfg.phoenix_dir == root.resolve() / "phoenix/convolved"
    assert not str(cfg.night_h5).startswith(str(elsewhere))


def test_tracked_manifest_unset_var_raises(monkeypatch) -> None:
    """An unset ``${TELLURICS_DATA}`` must not silently resolve to a path."""
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=DATA_ROOT_ENV):
        load_config(EXPERIMENTS / "multiple_frames" / "experiment1_smoke.yaml")


def test_data_config_rejects_nonexistent_root(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="data.root"):
        DataConfig(param_bounds=BOUNDS, root=tmp_path / "does-not-exist")
