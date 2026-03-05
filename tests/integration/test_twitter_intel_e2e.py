"""Integration smoke test for Twitter agent analyzer.

Uses REAL APIs (LLM + yfinance) to exercise the full PydanticAI agent
with tool calls. Market may be closed — options chain values can be 0/stale
but the pipeline (realized vol calc, expiry selection, chain formatting) is
validated end-to-end.

Run with: pytest -m integration tests/integration/test_twitter_intel_e2e.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from synesis.ingestion.twitterapi import Tweet
from synesis.processing.twitter.analyzer import TwitterAgentAnalyzer
from synesis.processing.twitter.models import TwitterAgentAnalysis
from synesis.providers.yfinance.client import YFinanceClient


def _make_tweet(
    username: str,
    text: str,
    hours_ago: float = 2.0,
) -> Tweet:
    return Tweet(
        tweet_id=f"integ_{username}_{hours_ago}",
        user_id=f"uid_{username}",
        username=username,
        text=text,
        timestamp=datetime.now(UTC) - timedelta(hours=hours_ago),
        raw={},
    )


# Realistic tweet batch that should trigger tool usage
SAMPLE_TWEETS = [
    _make_tweet(
        "aleabitoreddit",
        "NVDA looking incredibly strong into earnings next week. "
        "Datacenter revenue expected to blow past $40B. IV is surprisingly low "
        "given the magnitude of the catalyst. Calls look cheap.",
        hours_ago=3.0,
    ),
    _make_tweet(
        "unusual_whales",
        "Massive call sweep in AAPL Apr $250 calls, 10k contracts. "
        "Someone betting big on the iPhone 17 cycle. Premium paid: $2.3M.",
        hours_ago=2.5,
    ),
    _make_tweet(
        "NickTimiraos",
        "Fed officials signal patience on rate cuts, emphasizing data dependency. "
        "Core PCE still running above 2.5% target. Markets repricing to fewer cuts in 2026.",
        hours_ago=1.5,
    ),
    _make_tweet(
        "aleabitoreddit",
        "MU benefiting from DRAM supply tightening. Samsung cutting production. "
        "This is a multi-quarter story. Long MU, targeting $130.",
        hours_ago=1.0,
    ),
    _make_tweet(
        "zerohedge",
        "SPY breaking below 200d MA. Risk-off across the board. "
        "TLT catching a bid as flight to safety accelerates.",
        hours_ago=0.5,
    ),
]


@pytest.fixture
def yfinance_client(mock_redis: Any) -> YFinanceClient:
    """Real yfinance client with mock Redis (no API key needed, free data)."""
    return YFinanceClient(redis=mock_redis)


@pytest.mark.integration
class TestTwitterAgentE2E:
    """End-to-end test: real LLM + real yfinance, no mocks."""

    @pytest.mark.anyio
    async def test_full_analysis_produces_valid_output(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Full pipeline: tweets → LLM agent with tools → structured output."""
        analyzer = TwitterAgentAnalyzer()
        result = await analyzer.analyze_tweets(
            tweets=SAMPLE_TWEETS,
            yfinance=yfinance_client,
        )

        assert result is not None
        assert isinstance(result, TwitterAgentAnalysis)

        # Should have a market overview
        assert len(result.market_overview) > 20

        # Should have identified themes
        assert len(result.themes) >= 1

        # Should have set raw_tweet_count
        assert result.raw_tweet_count == len(SAMPLE_TWEETS)

        # Check themes have required fields
        for theme in result.themes:
            assert theme.title
            assert theme.summary
            assert theme.category in (
                "macro",
                "sector",
                "earnings",
                "geopolitical",
                "trade_idea",
                "technical",
            )
            assert len(theme.sources) >= 1
            assert theme.conviction in ("high", "medium", "low")

            # Check tickers have required fields
            for tm in theme.tickers:
                assert tm.ticker == tm.ticker.upper()
                assert tm.direction in ("bullish", "bearish", "neutral")
                assert len(tm.reasoning) > 5

    @pytest.mark.anyio
    async def test_at_least_one_ticker_has_price_context(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """LLM should call get_quote for individual stocks, filling price_context."""
        analyzer = TwitterAgentAnalyzer()
        result = await analyzer.analyze_tweets(
            tweets=SAMPLE_TWEETS,
            yfinance=yfinance_client,
        )

        assert result is not None

        # Collect all ticker mentions
        all_tickers = [tm for theme in result.themes for tm in theme.tickers]

        # At least one individual stock should have price context from get_quote
        tickers_with_price = [tm for tm in all_tickers if tm.price_context is not None]
        assert len(tickers_with_price) >= 1, (
            f"No tickers have price_context. Tickers: {[t.ticker for t in all_tickers]}"
        )

    @pytest.mark.anyio
    async def test_etfs_have_no_price_context(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """ETFs (SPY, TLT, QQQ) should NOT have price_context per prompt rules."""
        analyzer = TwitterAgentAnalyzer()
        result = await analyzer.analyze_tweets(
            tweets=SAMPLE_TWEETS,
            yfinance=yfinance_client,
        )

        assert result is not None

        etf_symbols = {"SPY", "QQQ", "TLT", "GLD", "USO", "UUP", "VIXY", "EEM",
                        "XLF", "XLK", "XLE", "XLV", "SMH", "IBB", "KRE"}
        for theme in result.themes:
            for tm in theme.tickers:
                if tm.ticker in etf_symbols:
                    assert tm.price_context is None, (
                        f"ETF {tm.ticker} should not have price_context"
                    )

    @pytest.mark.anyio
    async def test_analysis_without_yfinance_still_works(self) -> None:
        """Agent should degrade gracefully when yfinance is None."""
        analyzer = TwitterAgentAnalyzer()
        result = await analyzer.analyze_tweets(
            tweets=SAMPLE_TWEETS[:2],  # fewer tweets to save LLM cost
            yfinance=None,
        )

        assert result is not None
        assert isinstance(result, TwitterAgentAnalysis)
        assert len(result.market_overview) > 10


@pytest.mark.integration
class TestYFinanceToolsDirectly:
    """Validate the yfinance calls that back the agent tools.

    These run outside the LLM to confirm data quality and realized vol math.
    """

    @pytest.mark.anyio
    async def test_get_quote_returns_valid_data(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """get_quote should return a quote with at least ticker and last price."""
        q = await yfinance_client.get_quote("AAPL")
        assert q.ticker == "AAPL"
        # last can be None after hours but should generally be set
        # just check the object is well-formed
        assert q.name is not None
        assert len(q.name) > 0

    @pytest.mark.anyio
    async def test_get_history_returns_bars(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """1mo daily history should return enough bars for realized vol calc."""
        bars = await yfinance_client.get_history("AAPL", period="1mo", interval="1d")
        # Should have ~20-22 trading days in a month
        assert len(bars) >= 18, f"Expected ~20 bars for 1mo, got {len(bars)}"
        # Bars should have valid close prices
        closes = [b.close for b in bars if b.close is not None]
        assert len(closes) >= 18
        # Prices should be positive
        assert all(c > 0 for c in closes)

    @pytest.mark.anyio
    async def test_realized_vol_calculation(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Validate the realized vol math matches what the agent tool computes.

        The tool in get_options_snapshot computes:
        1. log returns from daily closes
        2. sample variance (ddof=1)
        3. annualize by sqrt(252)

        This test runs the same math on real data and checks it's reasonable.
        """
        import math

        bars = await yfinance_client.get_history("AAPL", period="1mo", interval="1d")
        closes = [b.close for b in bars if b.close is not None and b.close > 0]
        assert len(closes) >= 5, f"Need >=5 closes, got {len(closes)}"

        # Same math as the tool
        log_returns = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
        ]
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        realized_vol = math.sqrt(variance * 252)

        # Realized vol for AAPL should be somewhere between 5% and 150%
        # (extremely wide bounds — just sanity check the math isn't broken)
        assert 0.05 <= realized_vol <= 1.50, (
            f"Realized vol {realized_vol:.2%} outside sane range for AAPL"
        )

        # Verify it's not NaN or Inf
        assert math.isfinite(realized_vol)

    @pytest.mark.anyio
    async def test_options_expirations_available(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """AAPL should always have options expirations."""
        expirations = await yfinance_client.get_options_expirations("AAPL")
        assert len(expirations) >= 1
        # Should be ISO date strings
        for exp in expirations[:3]:
            assert len(exp) == 10  # YYYY-MM-DD
            assert exp[4] == "-"

    @pytest.mark.anyio
    async def test_options_chain_structure(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Options chain should have calls and puts with strikes."""
        expirations = await yfinance_client.get_options_expirations("AAPL")
        assert expirations, "No expirations found"

        # Pick the first expiry with at least 7 days out (same logic as tool)
        from datetime import date

        target_exp = expirations[0]
        today = date.today()
        for exp_str in expirations:
            exp_date = date.fromisoformat(exp_str)
            if (exp_date - today).days >= 7:
                target_exp = exp_str
                break

        chain = await yfinance_client.get_options_chain("AAPL", target_exp, greeks=True)

        # Should have both calls and puts
        assert len(chain.calls) > 0
        assert len(chain.puts) > 0

        # All contracts should have a strike
        for c in chain.calls[:5]:
            assert c.strike > 0
        for c in chain.puts[:5]:
            assert c.strike > 0

    @pytest.mark.anyio
    async def test_full_options_snapshot_flow(
        self,
        yfinance_client: YFinanceClient,
    ) -> None:
        """Simulate the full get_options_snapshot tool flow for AAPL.

        This mirrors the exact sequence of calls the tool makes:
        1. get_quote (spot price)
        2. get_history (1mo for realized vol)
        3. get_options_expirations
        4. get_options_chain

        Market may be closed — we just validate the pipeline doesn't error.
        """
        import math

        ticker = "AAPL"

        # 1. Quote
        quote = await yfinance_client.get_quote(ticker)
        spot = quote.last or 0
        assert spot >= 0  # 0 is ok if market closed

        # 2. Realized vol from 1mo history
        bars = await yfinance_client.get_history(ticker, period="1mo", interval="1d")
        closes = [b.close for b in bars if b.close is not None and b.close > 0]

        if len(closes) >= 5:
            log_returns = [
                math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            ]
            mean_r = sum(log_returns) / len(log_returns)
            variance = sum((r - mean_r) ** 2 for r in log_returns) / (
                len(log_returns) - 1
            )
            realized_vol = math.sqrt(variance * 252)
            assert math.isfinite(realized_vol)
            assert 0.01 <= realized_vol <= 3.0

        # 3. Expirations
        expirations = await yfinance_client.get_options_expirations(ticker)
        assert len(expirations) >= 1

        # 4. Chain for nearest valid expiry
        from datetime import date

        today = date.today()
        target_exp = expirations[0]
        for exp_str in expirations:
            exp_date = date.fromisoformat(exp_str)
            if (exp_date - today).days >= 7:
                target_exp = exp_str
                break

        chain = await yfinance_client.get_options_chain(ticker, target_exp, greeks=True)
        assert len(chain.calls) > 0 or len(chain.puts) > 0
