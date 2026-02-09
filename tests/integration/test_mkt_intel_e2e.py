"""Live integration smoke tests for Flow 3: Prediction Market Intelligence.

Tests with REAL Polymarket + Kalshi APIs (no API keys needed — public read).
Run with: pytest -m integration tests/integration/test_mkt_intel_e2e.py -v
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestMktIntelLiveSmoke:
    """Smoke test with REAL Polymarket + Kalshi APIs.

    Requires: No API keys needed (both are public read APIs).
    Run with: pytest -m integration
    """

    @pytest.mark.anyio
    async def test_polymarket_live(self) -> None:
        """Test live Polymarket API fetch."""
        from synesis.markets.polymarket import PolymarketClient

        async with PolymarketClient() as client:
            markets = await client.get_trending_markets(limit=5)

        print(f"\nPolymarket: {len(markets)} trending markets")
        for m in markets[:3]:
            print(f"  {m.question} @ ${m.yes_price:.2f}")

        assert len(markets) > 0

    @pytest.mark.anyio
    async def test_kalshi_live(self) -> None:
        """Test live Kalshi API fetch."""
        from synesis.markets.kalshi import KalshiClient

        async with KalshiClient() as client:
            markets = await client.get_markets(status="open", limit=5)

        print(f"\nKalshi: {len(markets)} open markets")
        for m in markets[:3]:
            print(f"  {m.title} @ ${m.yes_price:.2f}")

        assert len(markets) > 0

    @pytest.mark.anyio
    async def test_full_scan_live(self) -> None:
        """Test full scanner with live APIs but mocked storage."""
        from synesis.markets.kalshi import KalshiClient
        from synesis.markets.polymarket import PolymarketClient
        from synesis.processing.mkt_intel.scanner import MarketScanner

        async with PolymarketClient() as poly, KalshiClient() as kalshi:
            scanner = MarketScanner(
                polymarket=poly,
                kalshi=kalshi,
                ws_manager=None,
                db=None,
            )
            result = await scanner.scan()

        print("\nLive scan results:")
        print(f"  Total scanned: {result.total_markets_scanned}")
        print(f"  Trending: {len(result.trending_markets)}")
        print(f"  Expiring: {len(result.expiring_markets)}")

        for m in result.trending_markets[:3]:
            print(f"  [{m.platform}] {m.question} @ ${m.yes_price:.2f}")

        assert result.total_markets_scanned > 0
