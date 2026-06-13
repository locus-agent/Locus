"""Category inference: 'science' removed, 'crypto' expanded — with word-boundary safety."""
import pytest

from locus import config
from locus.markets.gamma import _infer_category


@pytest.mark.parametrize("question", [
    "Will Bitcoin be above $100k by July?",
    "Will BTC dominance exceed 60%?",
    "Will Ethereum complete the next upgrade in 2026?",
    "Will ETH flip below 0.05 BTC?",
    "Will Solana hit a new ATH?",
    "Will SOL be a top-3 chain by year end?",
    "Will a new memecoin 100x this month?",
    "Will Coinbase list a new altcoin?",
    "Will Binance face new charges?",
    "Will USDC depeg again?",
    "Will a major stablecoin lose its peg?",
    "Will a web3 startup IPO in 2026?",
    "Will NFT trading volume recover?",
    "Will a DeFi protocol be exploited?",
    "Will Polymarket hit record volume?",
])
def test_crypto_keywords_map_to_crypto(question):
    assert _infer_category(question, []) == "crypto"


@pytest.mark.parametrize("question", [
    "Will SpaceX launch Starship to orbit by 2026?",
    "Will NASA return to the Moon this decade?",
    "Will a major climate treaty pass?",
    "Will new research change the standard model?",
    "Will the discovery be confirmed?",
])
def test_former_science_questions_no_longer_science(question):
    cat = _infer_category(question, [])
    assert cat != "science"          # 'science' is gone entirely
    assert cat == "other"            # and these have no other category keyword


def test_science_not_tracked_crypto_is():
    assert "science" not in config.MARKET_CATEGORIES
    assert "crypto" in config.MARKET_CATEGORIES


@pytest.mark.parametrize("question,expected", [
    ("Will the new solar farm open?", "other"),   # 'sol' must not match in 'solar'
    ("Will ethics reform pass Congress?", "politics"),  # 'eth' not in 'ethics'; 'congress' wins
    ("Will the console launch on time?", "other"),  # 'sol' not in 'console'
])
def test_word_boundary_safety(question, expected):
    assert _infer_category(question, []) == expected


def test_ai_still_wins_over_crypto_ordering():
    # 'ai' is listed before 'crypto'; an AI+crypto question stays 'ai'.
    assert _infer_category("Will OpenAI launch a crypto token?", []) == "ai"
