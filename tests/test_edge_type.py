"""Edge-type classification: momentum detection, persistence, and breakdown."""
from locus.core import classifier
from locus.core.classifier import classify_edge_type, EDGE_TYPES
from locus.markets.gamma import Market

ALL_TIME = "2000-01-01 00:00:00"


def _mkt(condition_id="cond1", yes_price=0.50):
    return Market(condition_id, "Will X happen?", "ai", yes_price, 1.0 - yes_price,
                  5000, "", True, [])


def _log_class(db, condition_id="cond1", yes_price=0.50, action="signal",
               edge_type=None, direction="bullish"):
    return db.log_classification(
        market_question="Will X happen?", headline="h", news_source="rss",
        direction=direction, materiality=0.7, edge=0.2, action=action,
        condition_id=condition_id, yes_price=yes_price, edge_type=edge_type,
    )


# --- edge-type definition ---

def test_edge_types_defined_including_arbitrage():
    assert EDGE_TYPES == ("news", "momentum", "arbitrage")


# --- classify_edge_type ---

def test_defaults_to_news_with_no_history(tmp_db):
    assert classify_edge_type(_mkt()) == "news"


def test_small_move_is_news(tmp_db):
    _log_class(tmp_db, yes_price=0.50)        # baseline
    # current 0.52 -> 4% move, under the 10% momentum threshold
    assert classify_edge_type(_mkt(yes_price=0.52)) == "news"


def test_large_move_is_momentum(tmp_db):
    _log_class(tmp_db, yes_price=0.50)        # baseline
    # current 0.60 -> 20% relative move -> momentum
    assert classify_edge_type(_mkt(yes_price=0.60)) == "momentum"


def test_momentum_uses_earliest_price_in_window(tmp_db):
    # Oldest stored price is the baseline even when a later one is closer.
    _log_class(tmp_db, yes_price=0.40)        # earliest
    _log_class(tmp_db, yes_price=0.58)        # later, near current
    assert classify_edge_type(_mkt(yes_price=0.60)) == "momentum"  # vs 0.40


def test_other_market_history_does_not_count(tmp_db):
    _log_class(tmp_db, condition_id="other", yes_price=0.10)
    assert classify_edge_type(_mkt(condition_id="cond1", yes_price=0.60)) == "news"


# --- persistence ---

def test_log_trade_stores_edge_type(tmp_db):
    tmp_db.log_trade(
        market_id="cond1", market_question="Will X happen?", claude_score=0.7,
        market_price=0.5, edge=0.2, side="YES", amount_usd=25.0,
        status="dry_run", edge_type="momentum",
    )
    assert tmp_db.get_recent_trades(limit=1)[0]["edge_type"] == "momentum"


def test_log_classification_stores_edge_type(tmp_db):
    _log_class(tmp_db, edge_type="news")
    assert tmp_db.get_recent_classifications(limit=1)[0]["edge_type"] == "news"


def test_earliest_classification_price_helper(tmp_db):
    _log_class(tmp_db, yes_price=0.40)
    _log_class(tmp_db, yes_price=0.70)
    assert tmp_db.get_earliest_classification_price("cond1", 24.0) == 0.40
    assert tmp_db.get_earliest_classification_price("missing", 24.0) is None


# --- breakdown ---

def test_edge_type_breakdown_counts_signals_by_type(tmp_db):
    _log_class(tmp_db, action="signal", edge_type="news")
    _log_class(tmp_db, action="signal", edge_type="news")
    _log_class(tmp_db, action="signal", edge_type="momentum")
    _log_class(tmp_db, action="skip", edge_type="news")  # non-signal: excluded
    breakdown = tmp_db.get_edge_type_breakdown_since(ALL_TIME, action="signal")
    assert breakdown == {"news": 2, "momentum": 1}
