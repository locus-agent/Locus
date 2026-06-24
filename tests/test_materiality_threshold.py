"""Direction- and category-aware materiality floors.

Covers pipeline.get_materiality_threshold's priority order (category before
direction), the default fallback, the env-var overrides on the config constants,
and backward compat with the legacy MATERIALITY_THRESHOLD_BULLISH/_BEARISH and
SPORTS_MATERIALITY_THRESHOLD env vars.
"""
import importlib

import pytest

from locus import config
from locus.core.pipeline import get_materiality_threshold


@pytest.fixture(scope="module", autouse=True)
def _restore_config_after_module():
    """The env-override tests reload config with load_dotenv stubbed out. Reload
    it once more after this module (monkeypatch already torn down, so the real
    .env is read) so later test modules see the normal config."""
    yield
    importlib.reload(config)


@pytest.fixture(autouse=True)
def _pin_floors(monkeypatch):
    """Pin every floor to a distinct value so each branch is unambiguous."""
    monkeypatch.setattr(config, "MIN_MATERIALITY_DEFAULT", 0.33)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BULLISH", 0.34)
    monkeypatch.setattr(config, "MIN_MATERIALITY_BEARISH", 0.27)
    monkeypatch.setattr(config, "MIN_MATERIALITY_GEOPOLITICAL", 0.30)
    monkeypatch.setattr(config, "MIN_MATERIALITY_SPORTS", 0.40)


# --- Priority order ------------------------------------------------------

def test_geopolitical_category_wins():
    assert get_materiality_threshold("bullish", "geopolitical") == 0.30


def test_sports_category_wins():
    assert get_materiality_threshold("bearish", "sports") == 0.40


def test_category_beats_direction():
    # A bearish geopolitical call uses the geopolitical floor, not the bearish
    # one; a bullish sports call uses the sports floor, not the bullish one.
    assert get_materiality_threshold("bearish", "geopolitical") == 0.30
    assert get_materiality_threshold("bullish", "sports") == 0.40


def test_bearish_direction():
    assert get_materiality_threshold("bearish", "crypto") == 0.27


def test_bullish_direction():
    assert get_materiality_threshold("bullish", "ai") == 0.34


# --- Fallback ------------------------------------------------------------

def test_unknown_direction_and_category_falls_back_to_default():
    assert get_materiality_threshold("neutral", "other") == 0.33


def test_empty_inputs_fall_back_to_default():
    assert get_materiality_threshold("", "") == 0.33


def test_unknown_category_uses_direction():
    # An unrecognized category does not short-circuit; direction still applies.
    assert get_materiality_threshold("bearish", "") == 0.27
    assert get_materiality_threshold("bullish", "") == 0.34


# --- Env overrides + backward compat -------------------------------------

def _reload_config(monkeypatch, env):
    """Reload config with a clean materiality env so import-time os.getenv picks
    up exactly the provided overrides. Restores the module afterward.

    load_dotenv is stubbed to a no-op so the developer's .env (which may set the
    legacy materiality vars) doesn't leak into these isolation tests."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for var in (
        "MIN_MATERIALITY_DEFAULT", "MIN_MATERIALITY_BULLISH", "MIN_MATERIALITY_BEARISH",
        "MIN_MATERIALITY_GEOPOLITICAL", "MIN_MATERIALITY_SPORTS",
        "MATERIALITY_THRESHOLD_BULLISH", "MATERIALITY_THRESHOLD_BEARISH",
        "SPORTS_MATERIALITY_THRESHOLD",
    ):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(config)


def test_env_override_new_vars(monkeypatch):
    try:
        cfg = _reload_config(monkeypatch, {
            "MIN_MATERIALITY_DEFAULT": "0.11",
            "MIN_MATERIALITY_BULLISH": "0.22",
            "MIN_MATERIALITY_BEARISH": "0.12",
            "MIN_MATERIALITY_GEOPOLITICAL": "0.15",
            "MIN_MATERIALITY_SPORTS": "0.55",
        })
        assert cfg.MIN_MATERIALITY_DEFAULT == 0.11
        assert cfg.MIN_MATERIALITY_BULLISH == 0.22
        assert cfg.MIN_MATERIALITY_BEARISH == 0.12
        assert cfg.MIN_MATERIALITY_GEOPOLITICAL == 0.15
        assert cfg.MIN_MATERIALITY_SPORTS == 0.55
    finally:
        importlib.reload(config)


def test_defaults_when_unset(monkeypatch):
    try:
        cfg = _reload_config(monkeypatch, {})
        assert cfg.MIN_MATERIALITY_DEFAULT == 0.33
        assert cfg.MIN_MATERIALITY_BULLISH == 0.34
        assert cfg.MIN_MATERIALITY_BEARISH == 0.27
        assert cfg.MIN_MATERIALITY_GEOPOLITICAL == 0.30
        assert cfg.MIN_MATERIALITY_SPORTS == 0.40
    finally:
        importlib.reload(config)


def test_legacy_env_vars_used_as_fallback(monkeypatch):
    # Old env vars set, new ones unset -> the legacy values back the bullish/
    # bearish/sports floors.
    try:
        cfg = _reload_config(monkeypatch, {
            "MATERIALITY_THRESHOLD_BULLISH": "0.3",
            "MATERIALITY_THRESHOLD_BEARISH": "0.4",
            "SPORTS_MATERIALITY_THRESHOLD": "0.48",
        })
        assert cfg.MIN_MATERIALITY_BULLISH == 0.3
        assert cfg.MIN_MATERIALITY_BEARISH == 0.4
        assert cfg.MIN_MATERIALITY_SPORTS == 0.48
        # Untouched floors keep their new defaults.
        assert cfg.MIN_MATERIALITY_DEFAULT == 0.33
        assert cfg.MIN_MATERIALITY_GEOPOLITICAL == 0.30
    finally:
        importlib.reload(config)


def test_new_var_takes_precedence_over_legacy(monkeypatch):
    try:
        cfg = _reload_config(monkeypatch, {
            "MATERIALITY_THRESHOLD_BULLISH": "0.3",
            "MIN_MATERIALITY_BULLISH": "0.34",
        })
        assert cfg.MIN_MATERIALITY_BULLISH == 0.34
    finally:
        importlib.reload(config)
