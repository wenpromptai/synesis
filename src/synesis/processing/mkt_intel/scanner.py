"""Market scanner for Flow 3: Prediction Market Intelligence.

Scans Polymarket + Kalshi for trending markets, expiring markets,
volume spikes, and odds movements. Uses REST for discovery and
Redis for real-time data from WebSocket streams.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from synesis.core.constants import (
    CROSS_PLATFORM_ARB_MIN_GAP,
    CROSS_PLATFORM_MATCH_SIMILARITY,
)
from synesis.core.logging import get_logger
from synesis.markets.models import (
    CrossPlatformArb,
    OddsMovement,
    ScanResult,
    UnifiedMarket,
    VolumeSpike,
)

if TYPE_CHECKING:
    from model2vec import StaticModel

    from synesis.markets.kalshi import KalshiClient, KalshiMarket
    from synesis.markets.polymarket import PolymarketClient, SimpleMarket
    from synesis.markets.ws_manager import MarketWSManager
    from synesis.storage.database import Database

logger = get_logger(__name__)


def _poly_to_unified(m: SimpleMarket) -> UnifiedMarket:
    """Convert a Polymarket SimpleMarket to UnifiedMarket."""
    return UnifiedMarket(
        platform="polymarket",
        external_id=m.id,
        condition_id=m.condition_id,
        question=m.question,
        yes_price=m.yes_price,
        no_price=m.no_price,
        volume_24h=m.volume_24h,
        volume_total=m.volume_total,
        end_date=m.end_date,
        is_active=m.is_active,
        url=m.url,
        category=m.category,
        description=m.description,
        outcome_label=m.group_item_title,
        yes_outcome=m.yes_outcome,
        yes_token_id=m.yes_token_id,
    )


def _kalshi_to_unified(m: KalshiMarket) -> UnifiedMarket:
    """Convert a Kalshi market to UnifiedMarket."""
    return UnifiedMarket(
        platform="kalshi",
        external_id=m.ticker,
        ticker=m.ticker,
        question=m.title,
        yes_price=m.yes_price,
        no_price=m.no_price,
        volume_24h=float(m.volume_24h),
        volume_total=float(m.volume),
        open_interest=float(m.open_interest),
        end_date=m.close_time,
        is_active=m.is_active,
        url=m.url,
        category=m.category,
    )


class MarketScanner:
    """Scans Polymarket + Kalshi. Uses REST for discovery + Redis for real-time data."""

    def __init__(
        self,
        polymarket: PolymarketClient,
        kalshi: KalshiClient,
        ws_manager: MarketWSManager | None,
        db: Database | None,
        expiring_hours: int = 24,
        volume_spike_threshold: float = 1.0,
    ) -> None:
        self._polymarket = polymarket
        self._kalshi = kalshi
        self._ws_manager = ws_manager
        self._db = db
        self._expiring_hours = expiring_hours
        self._volume_spike_threshold = volume_spike_threshold
        # Cross-platform arb state
        self._model: StaticModel | None = None
        self._model_load_failed = False
        self._cross_platform_matches: list[tuple[UnifiedMarket, UnifiedMarket]] = []
        self._matches_version: int = 0

    @property
    def cross_platform_matches(self) -> list[tuple[UnifiedMarket, UnifiedMarket]]:
        """Matched cross-platform market pairs (Polymarket, Kalshi)."""
        return list(self._cross_platform_matches)

    @property
    def matches_version(self) -> int:
        """Increments when cross_platform_matches changes."""
        return self._matches_version

    def _get_model(self) -> StaticModel | None:
        """Lazy-load Model2Vec (same model as deduplication)."""
        if self._model is None and not self._model_load_failed:
            try:
                from model2vec import StaticModel

                self._model = StaticModel.from_pretrained("minishlab/potion-base-8M")
            except Exception as e:
                self._model_load_failed = True
                logger.error("Model download failed, cross-platform arb detection disabled", error=str(e))
        return self._model

    async def scan(self) -> ScanResult:
        """Run a full scan cycle.

        1. REST: Fetch trending + expiring from both platforms (parallel)
        2. Merge with real-time Redis data (WebSocket prices/volumes)
        3. Atomically capture + reset WS volume counters
        4. Detect odds movements (current vs previous snapshot)
        5. Detect volume spikes (captured volumes vs DB comparison)
        6. Update WebSocket subscriptions for next cycle
        7. Store snapshot to market_snapshots hypertable
        """
        now = datetime.now(UTC)
        log = logger.bind(scan_time=now.isoformat())
        log.debug("Market scan started")

        # 1. Fetch from both platforms in parallel
        (
            poly_trending,
            poly_expiring,
            kalshi_trending,
            kalshi_expiring,
        ) = await asyncio.gather(
            self._fetch_polymarket_trending(),
            self._fetch_polymarket_expiring(),
            self._fetch_kalshi_trending(),
            self._fetch_kalshi_expiring(),
        )

        # Filter out non-tradable markets (no volume, or settled prices)
        def _is_tradable(m: UnifiedMarket) -> bool:
            return m.volume_24h > 0 and 0.02 < m.yes_price < 0.98 and 0.02 < m.no_price < 0.98

        pre_filter_trending = poly_trending + kalshi_trending
        pre_filter_expiring = poly_expiring + kalshi_expiring
        all_trending = [m for m in pre_filter_trending if _is_tradable(m)]
        all_expiring = [m for m in pre_filter_expiring if _is_tradable(m)]
        filtered_count = (len(pre_filter_trending) - len(all_trending)) + (len(pre_filter_expiring) - len(all_expiring))
        if filtered_count:
            log.debug("Filtered non-tradable markets", removed=filtered_count)
        all_markets = {m.external_id: m for m in all_trending + all_expiring}

        # 2. Merge with real-time Redis data
        if self._ws_manager:
            for market in all_markets.values():
                market_key = market.yes_token_id or market.ticker or market.external_id
                rt_price = await self._ws_manager.get_realtime_price(market.platform, market_key)
                if rt_price:
                    market.yes_price = rt_price[0]
                    market.no_price = rt_price[1]

                rt_vol = await self._ws_manager.get_realtime_volume(market.platform, market_key)
                if rt_vol is not None and rt_vol > 0:
                    # Use real-time volume if available
                    market.volume_24h = max(market.volume_24h, rt_vol)

        total_scanned = len(all_markets)

        # 3. Atomically capture + reset WS volume counters
        captured_volumes: dict[str, float] = {}
        if self._ws_manager:
            for market in all_markets.values():
                market_key = market.yes_token_id or market.ticker or market.external_id
                try:
                    vol = await self._ws_manager.read_and_reset_volume(market.platform, market_key)
                    if vol is not None and vol > 0:
                        captured_volumes[market.external_id] = vol
                except Exception as e:
                    logger.warning(
                        "Volume capture failed",
                        market_id=market.external_id,
                        error=str(e),
                    )

        # 4. Detect odds movements
        try:
            odds_movements = await self._detect_odds_movements(list(all_markets.values()))
        except Exception as e:
            logger.error("Odds movement detection failed", error=str(e))
            odds_movements = []

        # 5. Detect volume spikes (captured volumes vs DB comparison)
        try:
            volume_spikes = await self._detect_volume_spikes(
                list(all_markets.values()), captured_volumes
            )
        except Exception as e:
            logger.error("Volume spike detection failed", error=str(e))
            volume_spikes = []

        # 6. Detect cross-platform arbitrage (use filtered markets only)
        try:
            filtered_poly = [m for m in all_trending + all_expiring if m.platform == "polymarket"]
            filtered_kalshi = [m for m in all_trending + all_expiring if m.platform == "kalshi"]
            cross_platform_arbs = self._detect_cross_platform_arbs(
                filtered_poly,
                filtered_kalshi,
            )
        except Exception as e:
            logger.error("Cross-platform arb detection failed", error=str(e))
            cross_platform_arbs = []

        # 7. Update WebSocket subscriptions
        if self._ws_manager:
            await self._ws_manager.update_subscriptions(list(all_markets.values()))

        # 8. Store snapshots (volumes already captured in step 3)
        await self._store_snapshots(list(all_markets.values()), captured_volumes)

        log.info(
            "Market scan complete",
            total_scanned=total_scanned,
            trending=len(all_trending),
            expiring=len(all_expiring),
            odds_movements=len(odds_movements),
            volume_spikes=len(volume_spikes),
            cross_platform_arbs=len(cross_platform_arbs),
        )

        return ScanResult(
            timestamp=now,
            trending_markets=all_trending,
            expiring_markets=all_expiring,
            odds_movements=odds_movements,
            volume_spikes=volume_spikes,
            cross_platform_arbs=cross_platform_arbs,
            total_markets_scanned=total_scanned,
        )

    # ─────────────────────────────────────────────────────────────
    # Platform Fetchers
    # ─────────────────────────────────────────────────────────────

    async def _fetch_polymarket_trending(self) -> list[UnifiedMarket]:
        """Fetch trending Polymarket markets."""
        try:
            markets = await self._polymarket.get_trending_markets(limit=30)
            return [_poly_to_unified(m) for m in markets]
        except Exception as e:
            logger.error("Polymarket trending fetch failed", error=str(e))
            return []

    async def _fetch_polymarket_expiring(self) -> list[UnifiedMarket]:
        """Fetch expiring Polymarket markets."""
        try:
            markets = await self._polymarket.get_expiring_markets(hours=self._expiring_hours)
            return [_poly_to_unified(m) for m in markets]
        except Exception as e:
            logger.error("Polymarket expiring fetch failed", error=str(e))
            return []

    async def _fetch_kalshi_trending(self) -> list[UnifiedMarket]:
        """Fetch trending Kalshi markets (by volume)."""
        try:
            markets = await self._kalshi.get_markets(status="open", limit=30)
            markets.sort(key=lambda m: m.volume_24h, reverse=True)
            unified = [_kalshi_to_unified(m) for m in markets[:30]]
        except Exception as e:
            logger.error("Kalshi trending fetch failed", error=str(e))
            return []

        try:
            await self._enrich_kalshi_categories(markets[:30], unified)
        except Exception as e:
            logger.warning("Kalshi trending category enrichment failed", error=str(e))
        return unified

    async def _fetch_kalshi_expiring(self) -> list[UnifiedMarket]:
        """Fetch expiring Kalshi markets."""
        try:
            markets = await self._kalshi.get_expiring_markets(hours=self._expiring_hours)
            unified = [_kalshi_to_unified(m) for m in markets]
        except Exception as e:
            logger.error("Kalshi expiring fetch failed", error=str(e))
            return []

        try:
            await self._enrich_kalshi_categories(markets, unified)
        except Exception as e:
            logger.warning("Kalshi expiring category enrichment failed", error=str(e))
        return unified

    async def _enrich_kalshi_categories(
        self,
        raw_markets: list[KalshiMarket],
        unified: list[UnifiedMarket],
    ) -> None:
        """Fetch event categories from Kalshi and apply to UnifiedMarket objects."""
        event_tickers = list({m.event_ticker for m in raw_markets if m.event_ticker})
        if not event_tickers:
            return

        categories = await self._kalshi.get_event_categories(event_tickers)

        # Build ticker → category map for raw markets
        ticker_to_event: dict[str, str] = {m.ticker: m.event_ticker for m in raw_markets}
        for market in unified:
            event_ticker = ticker_to_event.get(market.external_id, "")
            cat = categories.get(event_ticker)
            if cat and not market.category:
                market.category = cat

    # ─────────────────────────────────────────────────────────────
    # Detection Algorithms
    # ─────────────────────────────────────────────────────────────

    def _detect_cross_platform_arbs(
        self,
        poly_markets: list[UnifiedMarket],
        kalshi_markets: list[UnifiedMarket],
        min_gap: float = CROSS_PLATFORM_ARB_MIN_GAP,
        min_similarity: float = CROSS_PLATFORM_MATCH_SIMILARITY,
    ) -> list[CrossPlatformArb]:
        """Match Polymarket vs Kalshi questions by semantic similarity, find price gaps."""
        if not poly_markets or not kalshi_markets:
            return []

        model = self._get_model()
        if model is None:
            return []

        # Batch encode all questions
        poly_questions = [m.question for m in poly_markets]
        kalshi_questions = [m.question for m in kalshi_markets]

        poly_emb = model.encode(poly_questions)
        kalshi_emb = model.encode(kalshi_questions)

        # Normalize for cosine similarity
        poly_norm = poly_emb / (np.linalg.norm(poly_emb, axis=1, keepdims=True) + 1e-10)
        kalshi_norm = kalshi_emb / (np.linalg.norm(kalshi_emb, axis=1, keepdims=True) + 1e-10)

        # Pairwise cosine similarity matrix: (N_poly x N_kalshi)
        sim_matrix = poly_norm @ kalshi_norm.T

        # For each Poly market, find best Kalshi match
        arbs: list[CrossPlatformArb] = []
        matched_pairs: list[tuple[UnifiedMarket, UnifiedMarket]] = []

        # Sort all candidate pairs by similarity descending so the best match
        # always wins, preventing duplicate claims on the same market.
        candidates = []
        for i in range(len(poly_markets)):
            for j in range(len(kalshi_markets)):
                if float(sim_matrix[i, j]) >= min_similarity:
                    candidates.append((float(sim_matrix[i, j]), i, j))
        candidates.sort(reverse=True)

        matched_poly: set[int] = set()
        matched_kalshi: set[int] = set()

        for best_sim, i, j in candidates:
            if i in matched_poly or j in matched_kalshi:
                continue
            matched_poly.add(i)
            matched_kalshi.add(j)

            poly_mkt = poly_markets[i]
            kalshi_mkt = kalshi_markets[j]
            matched_pairs.append((poly_mkt, kalshi_mkt))

            price_gap = abs(poly_mkt.yes_price - kalshi_mkt.yes_price)
            if price_gap >= min_gap:
                # Determine which platform is cheaper for YES
                # v1: YES-side arbs only — NO is the complement
                if poly_mkt.yes_price < kalshi_mkt.yes_price:
                    buy_platform = "polymarket"
                else:
                    buy_platform = "kalshi"
                side = "yes"

                arbs.append(
                    CrossPlatformArb(
                        polymarket=poly_mkt,
                        kalshi=kalshi_mkt,
                        price_gap=price_gap,
                        suggested_buy_platform=buy_platform,
                        suggested_side=side,
                        match_similarity=best_sim,
                    )
                )

        # Store all matched pairs for real-time arb monitor
        self._cross_platform_matches = matched_pairs
        self._matches_version += 1

        arbs.sort(key=lambda a: a.price_gap, reverse=True)
        logger.debug(
            "Cross-platform arb scan",
            matched_pairs=len(matched_pairs),
            arbs_found=len(arbs),
        )
        return arbs

    async def _detect_odds_movements(self, markets: list[UnifiedMarket]) -> list[OddsMovement]:
        """Compare current prices vs previous snapshot (~55 min ago) from market_snapshots."""
        if not self._db or not markets:
            return []

        market_ids = [m.external_id for m in markets]
        market_by_id: dict[str, UnifiedMarket] = {m.external_id: m for m in markets}

        # Batch fetch: latest snapshot older than 50 min per market
        try:
            rows_1h = await self._db.fetch(
                """
                SELECT DISTINCT ON (market_external_id)
                    market_external_id, yes_price
                FROM market_snapshots
                WHERE market_external_id = ANY($1)
                  AND time < NOW() - INTERVAL '55 minutes'
                ORDER BY market_external_id, time DESC
                """,
                market_ids,
            )
        except Exception as e:
            logger.warning("Batch 1h snapshot fetch failed", error=str(e))
            return []

        prices_1h: dict[str, float] = {}
        for row in rows_1h:
            if row["yes_price"] is not None:
                prices_1h[row["market_external_id"]] = float(row["yes_price"])

        # Find markets with significant movement (>5 cents)
        significant_ids: list[str] = []
        for mid, prev_price in prices_1h.items():
            if abs(market_by_id[mid].yes_price - prev_price) >= 0.05:
                significant_ids.append(mid)

        # Batch fetch 6h snapshots only for significant movers
        prices_6h: dict[str, float] = {}
        if significant_ids:
            try:
                rows_6h = await self._db.fetch(
                    """
                    SELECT DISTINCT ON (market_external_id)
                        market_external_id, yes_price
                    FROM market_snapshots
                    WHERE market_external_id = ANY($1)
                      AND time < NOW() - INTERVAL '5 hours 55 minutes'
                    ORDER BY market_external_id, time DESC
                    """,
                    significant_ids,
                )
                for row in rows_6h:
                    if row["yes_price"] is not None:
                        prices_6h[row["market_external_id"]] = float(row["yes_price"])
            except Exception as e:
                logger.warning("Batch 6h snapshot fetch failed", error=str(e))

        # Build movements from significant movers
        movements: list[OddsMovement] = []
        for mid in significant_ids:
            market = market_by_id[mid]
            change_1h = market.yes_price - prices_1h[mid]
            change_6h = None
            if mid in prices_6h:
                change_6h = market.yes_price - prices_6h[mid]
            movements.append(
                OddsMovement(
                    market=market,
                    price_change_1h=change_1h,
                    price_change_6h=change_6h,
                    direction="up" if change_1h > 0 else "down",
                )
            )

        movements.sort(key=lambda m: abs(m.price_change_1h), reverse=True)
        return movements

    async def _detect_volume_spikes(
        self, markets: list[UnifiedMarket], captured_volumes: dict[str, float]
    ) -> list[VolumeSpike]:
        """Compare captured WS volumes vs previous snapshot to detect spikes.

        Args:
            markets: Current scan markets.
            captured_volumes: Atomically captured + reset volumes from scan().
        """
        if not captured_volumes or not self._db or not markets:
            return []

        # Batch fetch previous snapshot volumes (1 round-trip instead of N)
        market_ids = list(captured_volumes.keys())
        try:
            rows = await self._db.fetch(
                """
                SELECT DISTINCT ON (market_external_id)
                    market_external_id, volume_1h
                FROM market_snapshots
                WHERE market_external_id = ANY($1)
                  AND time < NOW() - INTERVAL '55 minutes'
                  AND volume_1h IS NOT NULL
                  AND volume_1h > 0
                ORDER BY market_external_id, time DESC
                """,
                market_ids,
            )
        except Exception as e:
            logger.warning("Volume spike batch query failed", error=str(e))
            return []

        prev_volumes: dict[str, float] = {
            row["market_external_id"]: float(row["volume_1h"]) for row in rows
        }

        market_by_id: dict[str, UnifiedMarket] = {m.external_id: m for m in markets}
        spikes: list[VolumeSpike] = []
        for mid, current_vol in captured_volumes.items():
            prev_vol = prev_volumes.get(mid)
            if prev_vol is None or prev_vol <= 0:
                continue
            pct_change = (current_vol - prev_vol) / prev_vol
            if pct_change >= self._volume_spike_threshold:
                spikes.append(
                    VolumeSpike(
                        market=market_by_id[mid],
                        volume_current=current_vol,
                        volume_previous=prev_vol,
                        pct_change=pct_change,
                    )
                )

        spikes.sort(key=lambda s: s.pct_change, reverse=True)
        return spikes

    # ─────────────────────────────────────────────────────────────
    # Storage
    # ─────────────────────────────────────────────────────────────

    async def _store_snapshots(
        self, markets: list[UnifiedMarket], captured_volumes: dict[str, float]
    ) -> None:
        """Write current state to market_snapshots.

        Volumes are already atomically captured + reset by scan() in step 3.
        """
        if not self._db:
            return

        now = datetime.now(UTC)
        for market in markets:
            try:
                ws_volume = captured_volumes.get(market.external_id)
                if ws_volume is not None and ws_volume > 0:
                    logger.debug(
                        "Volume snapshot stored",
                        market_id=market.external_id,
                        volume_1h=ws_volume,
                    )
                else:
                    ws_volume = None

                await self._db.insert_market_snapshot(
                    time=now,
                    platform=market.platform,
                    market_external_id=market.external_id,
                    question=market.question,
                    category=market.category,
                    yes_price=market.yes_price,
                    no_price=market.no_price,
                    volume_1h=ws_volume,
                    volume_24h=market.volume_24h if market.volume_24h > 0 else None,
                    volume_total=market.volume_total if market.volume_total > 0 else None,
                    trade_count_1h=None,
                    open_interest=market.open_interest,
                )
            except Exception as e:
                logger.warning(
                    "Snapshot storage failed",
                    market_id=market.external_id,
                    error=str(e),
                )
