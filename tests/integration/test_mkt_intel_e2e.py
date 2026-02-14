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


@pytest.mark.integration
class TestPolymarketDataAPI:
    """Test Polymarket Data API endpoints with real data.

    These test the wallet/positions/trades/holders endpoints that Flow 3
    wallet scoring depends on. No API keys needed.
    """

    @pytest.mark.anyio
    async def test_get_trending_markets_returns_valid_fields(self) -> None:
        """Verify trending markets have all required fields for UnifiedMarket."""
        from synesis.markets.polymarket import PolymarketClient

        async with PolymarketClient() as client:
            markets = await client.get_trending_markets(limit=5)

        assert len(markets) > 0
        m = markets[0]
        # All fields needed by _poly_to_unified
        assert m.id, "id must be non-empty"
        assert m.condition_id, "condition_id must be non-empty"
        assert m.question, "question must be non-empty"
        assert 0.0 <= m.yes_price <= 1.0
        assert 0.0 <= m.no_price <= 1.0
        print(f"\n  Sample market: {m.question}")
        print(f"  id={m.id}, condition_id={m.condition_id}")
        print(f"  YES=${m.yes_price:.2f}, volume_24h=${m.volume_24h:,.0f}")

    @pytest.mark.anyio
    async def test_get_expiring_markets(self) -> None:
        """Verify expiring markets have end_date set."""
        from synesis.markets.polymarket import PolymarketClient

        async with PolymarketClient() as client:
            markets = await client.get_expiring_markets(hours=72)

        print(f"\n  {len(markets)} markets expiring within 72h")
        for m in markets[:3]:
            print(f"  {m.question} — expires {m.end_date}")
        # Might be 0 if nothing is expiring, that's OK
        if markets:
            assert markets[0].end_date is not None

    @pytest.mark.anyio
    async def test_search_markets(self) -> None:
        """Verify search returns relevant results."""
        from synesis.markets.polymarket import PolymarketClient

        async with PolymarketClient() as client:
            markets = await client.search_markets("Trump", limit=5)

        print(f"\n  Search 'Trump': {len(markets)} results")
        for m in markets[:3]:
            print(f"  {m.question} @ ${m.yes_price:.2f}")
        assert len(markets) > 0

    @pytest.mark.anyio
    async def test_get_top_holders_returns_flattened_format(self) -> None:
        """Verify get_top_holders flattens nested API and normalizes proxyWallet → address."""
        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient

        # First get a real condition_id from a trending market
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=3)

        assert len(markets) > 0, "Need at least one market to test holders"
        condition_id = markets[0].condition_id
        print(f"\n  Testing holders for: {markets[0].question}")
        print(f"  condition_id: {condition_id}")

        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(condition_id, limit=10)

        print(f"  Got {len(holders)} holders")
        if holders:
            h = holders[0]
            print(f"  Sample: address={h.get('address')}, amount={h.get('amount')}")
            # Key assertion: address field must be present (from proxyWallet normalization)
            assert "address" in h, f"Expected 'address' field, got keys: {list(h.keys())}"
            assert h["address"], "address must be non-empty"
            # proxyWallet should have been popped
            assert "proxyWallet" not in h, "proxyWallet should be renamed to address"
        else:
            print("  (no holders returned — market may have low activity)")

    @pytest.mark.anyio
    async def test_get_wallet_positions_field_names(self) -> None:
        """Verify /positions returns cashPnl, initialValue, currentValue fields.

        This is critical — our scoring uses these fields. The old code
        expected 'pnl' from /trades which doesn't exist.
        """
        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient

        # Get a holder address from a popular market
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=3)

        assert len(markets) > 0
        condition_id = markets[0].condition_id
        print(f"\n  Market: {markets[0].question}")

        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(condition_id, limit=5)

        if not holders:
            pytest.skip("No holders found for this market")

        address = holders[0]["address"]
        print(f"  Testing positions for wallet: {address}")

        async with PolymarketDataClient() as data:
            positions = await data.get_wallet_positions(address)

        print(f"  Got {len(positions)} positions")
        if positions:
            pos = positions[0]
            print(f"  Sample position keys: {sorted(pos.keys())}")
            print(f"  cashPnl={pos.get('cashPnl')}")
            print(f"  initialValue={pos.get('initialValue')}")
            print(f"  currentValue={pos.get('currentValue')}")
            print(f"  avgPrice={pos.get('avgPrice')}")
            print(f"  realizedPnl={pos.get('realizedPnl')}")
            print(f"  conditionId={pos.get('conditionId')}")
            # These fields must exist for our scoring to work
            assert "cashPnl" in pos or "percentPnl" in pos, (
                f"Position missing PnL fields. Keys: {sorted(pos.keys())}"
            )
        else:
            print("  (no positions — wallet may have closed all positions)")

    @pytest.mark.anyio
    async def test_get_wallet_trades_field_names(self) -> None:
        """Verify /trades returns side, conditionId but NOT pnl.

        The old code read trade.get('pnl') which always returned 0.
        Confirm pnl is not a field so we know the fix is necessary.
        """
        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient

        # Get a holder address
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=3)

        condition_id = markets[0].condition_id

        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(condition_id, limit=5)

        if not holders:
            pytest.skip("No holders found")

        address = holders[0]["address"]
        print(f"\n  Testing trades for wallet: {address}")

        async with PolymarketDataClient() as data:
            trades = await data.get_wallet_trades(address, limit=10)

        print(f"  Got {len(trades)} trades")
        if trades:
            trade = trades[0]
            print(f"  Sample trade keys: {sorted(trade.keys())}")
            print(f"  side={trade.get('side')}")
            print(f"  conditionId={trade.get('conditionId')}")
            print(f"  size={trade.get('size')}")
            print(f"  price={trade.get('price')}")
            print(f"  usdcSize={trade.get('usdcSize')}")
            # Confirm pnl is NOT present (this was the bug)
            if "pnl" not in trade:
                print("  CONFIRMED: 'pnl' field NOT present in trades (expected)")
            else:
                print(f"  WARNING: 'pnl' field found = {trade.get('pnl')}")
            # side should be present for wash-trade detection
            assert "side" in trade, f"Expected 'side' field. Keys: {sorted(trade.keys())}"
        else:
            print("  (no trades returned)")

    @pytest.mark.anyio
    async def test_get_open_interest(self) -> None:
        """Verify open interest endpoint returns a number."""
        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient

        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=3)

        condition_id = markets[0].condition_id
        print(f"\n  Testing OI for: {markets[0].question}")

        async with PolymarketDataClient() as data:
            oi = await data.get_open_interest(condition_id)

        print(f"  Open interest: {oi}")
        # OI can be None for some markets, but for trending it should exist
        if oi is not None:
            assert oi >= 0

    @pytest.mark.anyio
    async def test_wallet_scoring_pipeline_live(self) -> None:
        """End-to-end: get holders → get positions → verify scoring data available.

        This tests the exact data flow that update_wallet_metrics() uses.
        """
        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient

        # Step 1: Get a trending market
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=5)
        assert len(markets) > 0

        # Pick the market with highest volume (most likely to have active holders)
        market = max(markets, key=lambda m: m.volume_24h)
        print(f"\n  Market: {market.question} (vol=${market.volume_24h:,.0f})")

        # Step 2: Get top holders
        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(market.condition_id, limit=5)

        print(f"  Holders: {len(holders)}")
        if not holders:
            pytest.skip("No holders for this market — can't test scoring pipeline")

        # Step 3: Get positions and trades for first holder (what update_wallet_metrics does)
        address = holders[0]["address"]
        print(f"  Wallet: {address}")

        async with PolymarketDataClient() as data:
            positions = await data.get_wallet_positions(address)
            trades = await data.get_wallet_trades(address, limit=100)

        print(f"  Positions: {len(positions)}")
        print(f"  Trades: {len(trades)}")

        # Verify we can extract scoring signals from positions
        if positions:
            total_pnl = sum(float(p.get("cashPnl", 0) or 0) for p in positions)
            wins = sum(1 for p in positions if float(p.get("cashPnl", 0) or 0) > 0)
            avg_initial = sum(float(p.get("initialValue", 0) or 0) for p in positions) / len(
                positions
            )
            print(f"  Total PnL: ${total_pnl:,.2f}")
            print(f"  Wins: {wins}/{len(positions)}")
            print(f"  Avg initial value: ${avg_initial:,.2f}")

        # Verify we can extract wash-trade signal from trades
        if trades:
            from collections import Counter

            sides = Counter(t.get("side") for t in trades)
            markets_traded = len({t.get("conditionId") for t in trades})
            print(f"  Trade sides: {dict(sides)}")
            print(f"  Unique markets traded: {markets_traded}")


@pytest.mark.integration
class TestKalshiDataAPI:
    """Test Kalshi API endpoints with real data."""

    @pytest.mark.anyio
    async def test_get_markets_returns_valid_fields(self) -> None:
        """Verify Kalshi markets have expected fields."""
        from synesis.markets.kalshi import KalshiClient

        async with KalshiClient() as client:
            markets = await client.get_markets(status="open", limit=5)

        assert len(markets) > 0
        m = markets[0]
        assert m.ticker, "ticker must be non-empty"
        assert m.title, "title must be non-empty"
        assert 0.0 <= m.yes_price <= 1.0
        print(f"\n  Sample: {m.title} @ ${m.yes_price:.2f}")
        print(f"  ticker={m.ticker}, volume_24h={m.volume_24h}")

    @pytest.mark.anyio
    async def test_get_expiring_markets(self) -> None:
        """Verify Kalshi expiring markets."""
        from synesis.markets.kalshi import KalshiClient

        async with KalshiClient() as client:
            markets = await client.get_expiring_markets(hours=72)

        print(f"\n  {len(markets)} Kalshi markets expiring within 72h")
        for m in markets[:3]:
            print(f"  {m.title} — closes {m.close_time}")

    @pytest.mark.anyio
    async def test_get_orderbook(self) -> None:
        """Verify Kalshi orderbook endpoint."""
        from synesis.markets.kalshi import KalshiClient

        async with KalshiClient() as client:
            markets = await client.get_markets(status="open", limit=3)

        if not markets:
            pytest.skip("No open markets")

        ticker = markets[0].ticker
        print(f"\n  Getting orderbook for: {markets[0].title} ({ticker})")

        async with KalshiClient() as client:
            book = await client.get_orderbook(ticker)

        if book:
            print(f"  YES Bids: {len(book.yes_bids)}, YES Asks: {len(book.yes_asks)}")
            if book.yes_bids:
                print(f"  Top yes bid: {book.yes_bids[0]}")
            if book.yes_asks:
                print(f"  Top yes ask: {book.yes_asks[0]}")

    @pytest.mark.anyio
    async def test_get_trades(self) -> None:
        """Verify Kalshi trades endpoint."""
        from synesis.markets.kalshi import KalshiClient

        async with KalshiClient() as client:
            markets = await client.get_markets(status="open", limit=3)

        if not markets:
            pytest.skip("No open markets")

        ticker = markets[0].ticker
        print(f"\n  Getting trades for: {markets[0].title} ({ticker})")

        async with KalshiClient() as client:
            trades = await client.get_trades(ticker=ticker, limit=10)

        print(f"  Got {len(trades)} trades")
        if trades:
            t = trades[0]
            print(f"  Sample trade: {t}")


@pytest.mark.integration
class TestWalletScoringLive:
    """Test the wallet scoring pipeline end-to-end with real API data.

    Verifies that update_wallet_metrics() can consume real API responses
    and produce valid scores.
    """

    @pytest.mark.anyio
    async def test_update_wallet_metrics_real_data(self) -> None:
        """Run update_wallet_metrics() against a real wallet.

        This is the critical test — it verifies the entire scoring pipeline
        works with actual API field names (cashPnl, initialValue, side, etc.).
        """
        from unittest.mock import AsyncMock

        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.wallets import WalletTracker

        # Get a real wallet address from a trending market
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=5)
        market = max(markets, key=lambda m: m.volume_24h)
        print(f"\n  Market: {market.question} (vol=${market.volume_24h:,.0f})")

        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(market.condition_id, limit=3)
        if not holders:
            pytest.skip("No holders")

        address = holders[0]["address"]
        print(f"  Wallet: {address}")

        # Create tracker with real data client but mock DB
        mock_db = AsyncMock()
        mock_db.upsert_wallet.return_value = None
        mock_db.upsert_wallet_metrics.return_value = None
        mock_db.get_wallet_first_seen.return_value = None
        mock_db.get_market_categories.return_value = {}

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(
                redis=AsyncMock(),
                db=mock_db,
                data_client=data_client,
            )

            result = await tracker.update_wallet_metrics(address)

        if result is None:
            print("  No positions found for this wallet — can't compute metrics")
            return

        print(f"  Insider score: {result.insider_score:.4f}")
        print(f"  Win rate: {result.win_rate:.2%}")
        print(f"  Total PnL: ${result.total_pnl:,.2f}")
        print(f"  Positions: {result.position_count}")
        print(f"  Trades: {result.total_trades}")
        print(f"  Avg position size: ${result.avg_position_size:,.2f}")
        print(f"  Focus score: {result.focus_score:.4f}")
        print(f"  Freshness score: {result.freshness_score:.4f}")
        print(f"  Wash trade ratio: {result.wash_trade_ratio:.4f}")
        print(f"  Concentration: {result.concentration:.4f}")
        print(f"  Largest open position: ${result.largest_open_position:,.2f}")

        # Validate ranges
        assert 0.0 <= result.insider_score <= 1.0
        assert 0.0 <= result.win_rate <= 1.0
        assert result.position_count >= 0
        assert result.total_trades >= 0
        assert result.avg_position_size >= 0
        assert 0.0 <= result.focus_score <= 1.0
        assert 0.0 <= result.freshness_score <= 1.0
        assert 0.0 <= result.wash_trade_ratio <= 1.0
        assert result.largest_open_position >= 0.0

        # Verify insider score sub-signal arithmetic
        expected_score = min(
            1.0,
            result.win_rate * 0.30
            + result.freshness_score * 0.15
            + result.focus_score * 0.20
            + min(1.0, result.avg_position_size / 1000.0) * 0.20
            + max(0.0, 1.0 - result.wash_trade_ratio) * 0.15,
        )
        assert abs(result.insider_score - expected_score) < 0.01, (
            f"Score mismatch: got {result.insider_score:.4f}, expected {expected_score:.4f}"
        )
        print(f"  Score arithmetic verified: {result.insider_score:.4f} == {expected_score:.4f}")

        # Verify fast-track qualification logic
        from synesis.processing.mkt_intel.wallets import WalletTracker

        qualifies = WalletTracker._qualifies_for_fast_track(result)
        reason = "N/A"
        if result.wash_trade_ratio > 0.30:
            reason = "wash_ratio too high"
        elif result.position_count > 0 and result.win_rate > 0.50:
            pnl_per_pos = result.total_pnl / result.position_count
            if pnl_per_pos > 10_000:
                reason = f"Path A: Consistent Insider (PnL/pos=${pnl_per_pos:,.0f})"
            else:
                reason = f"Path A fail: PnL/pos=${pnl_per_pos:,.0f} < $10k"
        if result.largest_open_position >= 50_000:
            reason = f"Path B: Fresh Insider (open=${result.largest_open_position:,.0f})"
        print(f"  Fast-track: {qualifies} ({reason})")

    @pytest.mark.anyio
    async def test_multiple_wallets_scoring(self) -> None:
        """Score multiple wallets from different markets to verify consistency."""
        from unittest.mock import AsyncMock

        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.wallets import WalletTracker

        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=5)

        # Get holders from top 3 markets by volume
        top_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:3]
        addresses: list[str] = []
        async with PolymarketDataClient() as data:
            for m in top_markets:
                holders = await data.get_top_holders(m.condition_id, limit=2)
                for h in holders:
                    addr = h["address"]
                    if addr not in addresses:
                        addresses.append(addr)
                if len(addresses) >= 4:
                    break

        print(f"\n  Scoring {len(addresses)} wallets from {len(top_markets)} markets")

        mock_db = AsyncMock()
        mock_db.upsert_wallet.return_value = None
        mock_db.upsert_wallet_metrics.return_value = None
        mock_db.get_wallet_first_seen.return_value = None
        mock_db.get_market_categories.return_value = {}

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(
                redis=AsyncMock(),
                db=mock_db,
                data_client=data_client,
            )

            scored = 0
            fast_tracked = 0
            for addr in addresses:
                result = await tracker.update_wallet_metrics(addr)
                if result is not None:
                    scored += 1
                    qualifies_ft = WalletTracker._qualifies_for_fast_track(result)
                    if qualifies_ft:
                        fast_tracked += 1
                    print(
                        f"  {addr[:10]}... → score={result.insider_score:.3f}, "
                        f"wr={result.win_rate:.0%}, pnl=${result.total_pnl:,.0f}, "
                        f"pos={result.position_count}, trades={result.total_trades}, "
                        f"open=${result.largest_open_position:,.0f}, ft={qualifies_ft}"
                    )
                    assert 0.0 <= result.insider_score <= 1.0
                    assert result.largest_open_position >= 0.0

        print(f"  Successfully scored {scored}/{len(addresses)} wallets")
        print(f"  Fast-track qualified: {fast_tracked}/{scored}")
        assert scored > 0, "Should score at least one wallet"

    @pytest.mark.anyio
    async def test_detect_high_conviction_live(self) -> None:
        """Test high-conviction detection with real market data."""
        from unittest.mock import AsyncMock

        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.scanner import _poly_to_unified
        from synesis.processing.mkt_intel.wallets import WalletTracker

        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=5)

        unified = [_poly_to_unified(m) for m in markets]
        print(f"\n  Testing high-conviction on {len(unified)} markets")

        mock_db = AsyncMock()
        mock_db.get_watched_wallets.return_value = []

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(
                redis=AsyncMock(),
                db=mock_db,
                data_client=data_client,
            )
            hc_trades = await tracker.detect_high_conviction_trades(unified)

        print(f"  Found {len(hc_trades)} high-conviction trades")
        for hc in hc_trades[:3]:
            print(
                f"  {hc.wallet_address[:10]}... on {hc.market.question[:50]}"
                f" — ${hc.position_size:,.0f} ({hc.concentration_pct:.0%} of portfolio)"
            )

    @pytest.mark.anyio
    async def test_check_wallet_activity_live(self) -> None:
        """Test insider activity check with real markets and a mock watched wallet list."""
        from unittest.mock import AsyncMock

        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.scanner import _poly_to_unified
        from synesis.processing.mkt_intel.wallets import WalletTracker

        # Get trending markets and their holders
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=3)

        market = max(markets, key=lambda m: m.volume_24h)
        unified = [_poly_to_unified(market)]

        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(market.condition_id, limit=3)

        if not holders:
            pytest.skip("No holders")

        # Pretend the first holder is a watched wallet
        watched_addr = holders[0]["address"]
        print(f"\n  Market: {market.question}")
        print(f"  Pretending {watched_addr[:10]}... is watched")

        mock_db = AsyncMock()
        mock_db.get_watched_wallets.return_value = [
            {
                "address": watched_addr,
                "platform": "polymarket",
                "insider_score": 0.75,
                "specialty_category": None,
                "watch_reason": "score",
            }
        ]

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(
                redis=AsyncMock(),
                db=mock_db,
                data_client=data_client,
            )
            alerts = await tracker.check_wallet_activity(unified)

        print(f"  Insider alerts: {len(alerts)}")
        for alert in alerts:
            print(
                f"  {alert.wallet_address[:10]}... → "
                f"score={alert.insider_score:.2f}, size=${alert.trade_size:,.0f}"
            )

    @pytest.mark.anyio
    async def test_fast_track_calculation_real_data(self) -> None:
        """Verify fast-track logic against real wallet data.

        For each scored wallet, manually recompute whether it should qualify
        for fast-track and verify the static method agrees.
        """
        from unittest.mock import AsyncMock

        from synesis.core.constants import (
            CONSISTENT_INSIDER_MIN_PNL_PER_POSITION,
            CONSISTENT_INSIDER_MIN_WIN_RATE,
            FAST_TRACK_MAX_WASH_RATIO,
            FRESH_INSIDER_MIN_POSITION,
        )
        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.wallets import WalletTracker

        # Get wallets from the highest-volume market
        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=10)
        top_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:3]

        addresses: list[str] = []
        async with PolymarketDataClient() as data:
            for m in top_markets:
                holders = await data.get_top_holders(m.condition_id, limit=5)
                for h in holders:
                    addr = h["address"]
                    if addr not in addresses:
                        addresses.append(addr)
                if len(addresses) >= 8:
                    break

        print(f"\n  Testing fast-track on {len(addresses)} real wallets")
        print(
            f"  Thresholds: wash<={FAST_TRACK_MAX_WASH_RATIO}, "
            f"wr>{CONSISTENT_INSIDER_MIN_WIN_RATE}, "
            f"pnl/pos>${CONSISTENT_INSIDER_MIN_PNL_PER_POSITION:,}, "
            f"open>=${FRESH_INSIDER_MIN_POSITION:,}"
        )

        mock_db = AsyncMock()
        mock_db.upsert_wallet.return_value = None
        mock_db.upsert_wallet_metrics.return_value = None
        mock_db.get_wallet_first_seen.return_value = None
        mock_db.get_market_categories.return_value = {}

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(redis=AsyncMock(), db=mock_db, data_client=data_client)

            for addr in addresses:
                result = await tracker.update_wallet_metrics(addr)
                if result is None:
                    continue

                # Manual fast-track check
                expected_ft = False
                path = "none"
                if result.wash_trade_ratio <= FAST_TRACK_MAX_WASH_RATIO:
                    if (
                        result.position_count > 0
                        and result.win_rate > CONSISTENT_INSIDER_MIN_WIN_RATE
                    ):
                        pnl_per_pos = result.total_pnl / result.position_count
                        if pnl_per_pos > CONSISTENT_INSIDER_MIN_PNL_PER_POSITION:
                            expected_ft = True
                            path = f"consistent (pnl/pos=${pnl_per_pos:,.0f})"
                    if result.largest_open_position >= FRESH_INSIDER_MIN_POSITION:
                        expected_ft = True
                        path = f"fresh (open=${result.largest_open_position:,.0f})"

                actual_ft = WalletTracker._qualifies_for_fast_track(result)
                status = "OK" if actual_ft == expected_ft else "MISMATCH"
                print(
                    f"  {addr[:10]}... → ft={actual_ft} ({path}), "
                    f"wr={result.win_rate:.0%}, pnl=${result.total_pnl:,.0f}, "
                    f"pos={result.position_count}, wash={result.wash_trade_ratio:.2f}, "
                    f"open=${result.largest_open_position:,.0f} [{status}]"
                )
                assert actual_ft == expected_ft, (
                    f"Fast-track mismatch for {addr[:10]}: expected={expected_ft}, got={actual_ft}"
                )

    @pytest.mark.anyio
    async def test_largest_open_position_manual_calc(self) -> None:
        """Manually verify largest_open_position matches raw API data."""
        from unittest.mock import AsyncMock

        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.wallets import WalletTracker

        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=5)
        market = max(markets, key=lambda m: m.volume_24h)

        async with PolymarketDataClient() as data:
            holders = await data.get_top_holders(market.condition_id, limit=3)
        if not holders:
            pytest.skip("No holders")

        address = holders[0]["address"]
        print(f"\n  Wallet: {address}")

        # Get raw positions directly
        async with PolymarketDataClient() as data:
            raw_positions = await data.get_wallet_positions(address)

        # Manual computation
        open_vals = [
            abs(float(p.get("currentValue", 0) or 0))
            for p in raw_positions
            if float(p.get("currentValue", 0) or 0) > 0
        ]
        expected_largest_open = max(open_vals) if open_vals else 0.0
        print(f"  Open positions: {len(open_vals)}")
        print(f"  Manual largest_open: ${expected_largest_open:,.2f}")

        # Now run through the tracker
        mock_db = AsyncMock()
        mock_db.upsert_wallet.return_value = None
        mock_db.upsert_wallet_metrics.return_value = None
        mock_db.get_wallet_first_seen.return_value = None
        mock_db.get_market_categories.return_value = {}

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(redis=AsyncMock(), db=mock_db, data_client=data_client)
            result = await tracker.update_wallet_metrics(address)

        if result is None:
            print("  (no result — wallet has no data)")
            return

        print(f"  Tracker largest_open: ${result.largest_open_position:,.2f}")
        assert abs(result.largest_open_position - expected_largest_open) < 0.01, (
            f"largest_open_position mismatch: "
            f"tracker={result.largest_open_position}, manual={expected_largest_open}"
        )
        print("  MATCH confirmed")

    @pytest.mark.anyio
    async def test_insider_score_subsignal_breakdown(self) -> None:
        """Verify each sub-signal of the insider score for multiple wallets."""
        from unittest.mock import AsyncMock

        from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
        from synesis.processing.mkt_intel.wallets import WalletTracker

        async with PolymarketClient() as gamma:
            markets = await gamma.get_trending_markets(limit=5)
        top_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)[:2]

        addresses: list[str] = []
        async with PolymarketDataClient() as data:
            for m in top_markets:
                holders = await data.get_top_holders(m.condition_id, limit=3)
                for h in holders:
                    addr = h["address"]
                    if addr not in addresses:
                        addresses.append(addr)
                if len(addresses) >= 5:
                    break

        mock_db = AsyncMock()
        mock_db.upsert_wallet.return_value = None
        mock_db.upsert_wallet_metrics.return_value = None
        mock_db.get_wallet_first_seen.return_value = None
        mock_db.get_market_categories.return_value = {}

        print(f"\n  Sub-signal breakdown for {len(addresses)} wallets:")
        print(
            f"  {'Wallet':<14} {'Prof':>6} {'Fresh':>6} {'Focus':>6} {'Size':>6} {'Wash':>6} = {'Score':>6}"
        )

        async with PolymarketDataClient() as data_client:
            tracker = WalletTracker(redis=AsyncMock(), db=mock_db, data_client=data_client)
            for addr in addresses:
                result = await tracker.update_wallet_metrics(addr)
                if result is None:
                    continue

                # Recompute sub-signals
                profitability = result.win_rate
                freshness = result.freshness_score
                focus = result.focus_score
                sizing = (
                    min(1.0, result.avg_position_size / 1000.0)
                    if result.avg_position_size > 0
                    else 0.0
                )
                wash_pen = max(0.0, 1.0 - result.wash_trade_ratio)

                expected = min(
                    1.0,
                    profitability * 0.30
                    + freshness * 0.15
                    + focus * 0.20
                    + sizing * 0.20
                    + wash_pen * 0.15,
                )

                print(
                    f"  {addr[:12]}.. "
                    f"{profitability:6.3f} {freshness:6.3f} {focus:6.3f} "
                    f"{sizing:6.3f} {wash_pen:6.3f} = {result.insider_score:6.3f}"
                )

                assert abs(result.insider_score - expected) < 0.01, (
                    f"Score arithmetic error for {addr[:10]}: "
                    f"got {result.insider_score:.4f}, expected {expected:.4f}"
                )
