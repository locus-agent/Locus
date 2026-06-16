"""DAILY_SPEND_LIMIT_USD gate: caps notional deployed per day, enforced in
dry-run mode (where rows are logged with status 'dry_run', never 'filled')."""
from locus import config
from locus.core import executor
from locus.markets.gamma import Market


def _signal(bet_amount: float):
    market = Market(
        condition_id="0xabc",
        question="Will it rain tomorrow?",
        category="weather",
        yes_price=0.5,
        no_price=0.5,
        volume=10_000.0,
        end_date="2026-12-31",
        active=True,
        tokens=[],
    )
    return executor.Signal(
        market=market,
        claude_score=0.7,
        market_price=0.5,
        edge=0.2,
        side="YES",
        bet_amount=bet_amount,
        reasoning="test",
        headlines="test headline",
    )


def test_limit_triggers_in_dry_run(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 50.0)

    # First two $25 dry-run trades fill the $50/day budget.
    assert executor.execute_trade(_signal(25.0))["status"] == "dry_run"
    assert executor.execute_trade(_signal(25.0))["status"] == "dry_run"

    # The third would push notional to $75 > $50: rejected, even in dry-run.
    assert executor.execute_trade(_signal(25.0))["status"] == "rejected_daily_limit"


def test_below_limit_does_not_trigger(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 100.0)

    for _ in range(3):
        assert executor.execute_trade(_signal(25.0))["status"] == "dry_run"

    # $75 deployed, still under the $100 cap.
    assert abs(tmp_db.get_daily_pnl()) == 75.0
