"""Multi-time-horizon classification: parsing the time_horizon field and the
horizon-penalized adjusted_materiality the gates judge signals by."""
import pytest

from locus import config
from locus.core.classifier import Classification, parse_classification


def _cls(materiality, time_horizon="medium"):
    return Classification(
        direction="bullish", materiality=materiality, confidence=0.6,
        reasoning="", latency_ms=1, model="test", time_horizon=time_horizon,
    )


@pytest.fixture(autouse=True)
def _pin_penalties(monkeypatch):
    # Pin to the documented defaults so the suite doesn't depend on a .env override.
    monkeypatch.setattr(
        config, "TIME_HORIZON_PENALTY",
        {"immediate": 0.0, "medium": -0.03, "long_term": -0.10},
    )


# --- parsing -----------------------------------------------------------------

def test_parser_extracts_time_horizon_from_json():
    out = parse_classification(
        '{"direction":"bullish","materiality":0.7,"confidence":0.8,'
        '"time_horizon":"long_term","reasoning":"r"}'
    )
    assert out["time_horizon"] == "long_term"


def test_parser_defaults_time_horizon_to_medium_when_absent():
    out = parse_classification('{"direction":"bullish","materiality":0.5,"confidence":0.7}')
    assert out["time_horizon"] == "medium"


def test_parser_normalizes_unknown_horizon_to_medium():
    out = parse_classification(
        '{"direction":"bullish","materiality":0.5,"time_horizon":"someday"}'
    )
    assert out["time_horizon"] == "medium"


def test_parser_extracts_time_horizon_from_regex_fallback():
    text = "direction: bullish, materiality: 0.5, time_horizon: immediate — reasoning"
    out = parse_classification(text)
    assert out["time_horizon"] == "immediate"


# --- adjusted_materiality penalty --------------------------------------------

def test_immediate_horizon_has_no_penalty():
    assert _cls(0.50, "immediate").adjusted_materiality == pytest.approx(0.50)


def test_medium_horizon_small_penalty():
    assert _cls(0.50, "medium").adjusted_materiality == pytest.approx(0.47)


def test_long_term_horizon_full_penalty():
    assert _cls(0.50, "long_term").adjusted_materiality == pytest.approx(0.40)


def test_strong_long_term_signal_still_clears_floor():
    # A strong (0.60) long-term call keeps 0.50 adjusted — still well above the
    # bullish floor (0.3), so it can still trade (the soft-penalty intent).
    assert _cls(0.60, "long_term").adjusted_materiality == pytest.approx(0.50)


def test_weak_long_term_signal_floored_at_zero():
    # 0.05 - 0.10 would be negative; max(0.0, ...) clamps to 0.0.
    assert _cls(0.05, "long_term").adjusted_materiality == 0.0


def test_unknown_horizon_uses_medium_default_penalty():
    # An unnormalized label on the dataclass falls back to the -0.03 default.
    assert _cls(0.50, "bogus").adjusted_materiality == pytest.approx(0.47)
