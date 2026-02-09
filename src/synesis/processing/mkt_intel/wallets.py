"""Wallet intelligence tracker for Flow 3 (Polymarket only).

Tracks whale/insider wallets, monitors their trading activity,
calculates insider scores, and generates alerts when watched
wallets trade on markets we're scanning.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from synesis.core.constants import MARKET_INTEL_REDIS_PREFIX
from synesis.core.logging import get_logger
from synesis.markets.models import InsiderAlert, UnifiedMarket

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.markets.polymarket import PolymarketDataClient
    from synesis.storage.database import Database

logger = get_logger(__name__)

# Rate limiting delay between API calls (100ms)
_API_DELAY_SECONDS = 0.1

_WALLETS_KEY = f"{MARKET_INTEL_REDIS_PREFIX}:wallets:watched"
_SCORE_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:wallets:score"
_SCORE_TTL = 3600  # 1 hour cache


class WalletTracker:
    """Tracks whale/insider wallets on Polymarket.

    Maintains a set of watched wallets (stored in DB + cached in Redis),
    checks their activity against scanned markets, and updates metrics.
    """

    def __init__(
        self,
        redis: Redis,
        db: Database | None = None,
        data_client: PolymarketDataClient | None = None,
        insider_score_min: float = 0.5,
    ) -> None:
        self._redis = redis
        self._db = db
        self._data_client = data_client
        self._insider_score_min = insider_score_min

    async def get_watched_wallets(self) -> list[dict[str, str | float]]:
        """Get watched wallets from DB (no caching).

        Returns:
            List of dicts with 'address', 'platform', 'insider_score'
        """
        if not self._db:
            return []

        try:
            wallets = await self._db.get_watched_wallets("polymarket")
            return [
                {
                    "address": w["address"],
                    "platform": w["platform"],
                    "insider_score": float(w.get("insider_score", 0) or 0),
                }
                for w in wallets
            ]
        except Exception as e:
            logger.error("Failed to get watched wallets", error=str(e))
            return []

    async def check_wallet_activity(self, markets: list[UnifiedMarket]) -> list[InsiderAlert]:
        """Cross-reference top holders of scanned markets with watched wallets.

        For each Polymarket market, fetches top holders and checks if any
        are in our watched wallet set.

        Args:
            markets: Markets from the current scan

        Returns:
            List of insider activity alerts
        """
        if not self._data_client or not self._db:
            return []

        watched = await self.get_watched_wallets()
        if not watched:
            return []

        watched_addresses = {str(w["address"]).lower() for w in watched}
        watched_scores = {str(w["address"]).lower(): float(w["insider_score"]) for w in watched}

        alerts: list[InsiderAlert] = []

        # Only check Polymarket markets (Kalshi has no public wallet data)
        poly_markets = [m for m in markets if m.platform == "polymarket" and m.condition_id]

        for market in poly_markets[:20]:  # Limit API calls
            try:
                holders = await self._data_client.get_top_holders(
                    market.condition_id or "", limit=20
                )
                for holder in holders:
                    address = str(holder.get("address", "")).lower()
                    if address in watched_addresses:
                        score = watched_scores.get(address, 0)
                        if score >= self._insider_score_min:
                            # Determine trade direction from position
                            position_size = float(holder.get("amount", 0))
                            outcome = holder.get("outcome", "")

                            alerts.append(
                                InsiderAlert(
                                    market=market,
                                    wallet_address=address,
                                    insider_score=score,
                                    trade_direction=outcome or "unknown",
                                    trade_size=abs(position_size),
                                )
                            )
            except Exception as e:
                logger.warning(
                    "Wallet activity check failed",
                    market_id=market.external_id,
                    error=str(e),
                )

        if alerts:
            logger.info(
                "Insider activity detected",
                alert_count=len(alerts),
                unique_wallets=len({a.wallet_address for a in alerts}),
            )

        return alerts

    async def discover_wallets_from_market(self, condition_id: str) -> list[str]:
        """Find new wallets worth watching from a market's top holders.

        Examines top holders and identifies wallets with profitable
        historical trading patterns. Newly discovered wallets are
        stored in the DB with is_watched=False (manual review needed).

        Args:
            condition_id: Polymarket condition ID

        Returns:
            List of newly discovered wallet addresses
        """
        if not self._data_client or not self._db:
            return []

        discovered: list[str] = []
        try:
            holders = await self._data_client.get_top_holders(condition_id, limit=20)
            for holder in holders:
                address = holder.get("address", "")
                if not address:
                    continue

                # Check if we already track this wallet
                existing = await self._db.fetchrow(
                    "SELECT id FROM wallets WHERE platform = 'polymarket' AND address = $1",
                    address,
                )
                if existing:
                    continue

                # Store as new wallet (not yet watched)
                await self._db.upsert_wallet(address, "polymarket")
                discovered.append(address)

            if discovered:
                logger.info(
                    "New wallets discovered",
                    count=len(discovered),
                    condition_id=condition_id,
                )
        except Exception as e:
            logger.error(
                "Wallet discovery failed",
                condition_id=condition_id,
                error=str(e),
            )

        return discovered

    async def update_wallet_metrics(self, address: str) -> tuple[float, int]:
        """Refresh wallet metrics from trade history.

        Args:
            address: Wallet address to update

        Returns:
            Tuple of (insider_score, total_trades) or (0.0, 0) on failure
        """
        if not self._data_client or not self._db:
            return 0.0, 0

        try:
            trades = await self._data_client.get_wallet_trades(address, limit=500)
            if not trades:
                return 0.0, 0

            total = len(trades)
            wins = 0
            total_pnl = 0.0

            for trade in trades:
                pnl = float(trade.get("pnl", 0) or 0)
                total_pnl += pnl
                if pnl > 0:
                    wins += 1

            win_rate = wins / total if total > 0 else 0.0

            # Simple insider score: win_rate * log(total_trades)
            insider_score = min(1.0, win_rate * math.log10(max(total, 1)) / 2)

            await self._db.upsert_wallet_metrics(
                address=address,
                platform="polymarket",
                total_trades=total,
                wins=wins,
                win_rate=win_rate,
                total_pnl=total_pnl,
                insider_score=insider_score,
            )
            logger.debug(
                "Wallet metrics updated",
                address=address[:10] + "...",
                total_trades=total,
                win_rate=f"{win_rate:.2f}",
                insider_score=f"{insider_score:.2f}",
            )
            return insider_score, total
        except Exception as e:
            logger.error(
                "Wallet metrics update failed",
                address=address[:10] + "...",
                error=str(e),
            )
            return 0.0, 0

    async def discover_and_score_wallets(
        self,
        markets: list[UnifiedMarket],
        top_n_markets: int = 5,
        auto_watch_threshold: float = 0.6,
        min_trades_to_watch: int = 20,
    ) -> int:
        """Discover wallets from top holders and auto-watch high scorers.

        This method:
        1. Takes the top N markets by volume
        2. Fetches top 20 holders for each market (Polymarket only)
        3. Upserts each wallet address to the DB
        4. Skips wallets already scored within the last 24 hours
        5. Fetches trade history and calculates insider score
        6. Auto-watches wallets meeting the threshold criteria

        Args:
            markets: Markets from the current scan
            top_n_markets: Number of top markets to process
            auto_watch_threshold: Minimum insider score to auto-watch
            min_trades_to_watch: Minimum trades required to auto-watch

        Returns:
            Number of newly watched wallets
        """
        if not self._data_client or not self._db:
            return 0

        log = logger.bind(
            top_n_markets=top_n_markets,
            auto_watch_threshold=auto_watch_threshold,
        )
        log.info("Starting wallet discovery pipeline")

        # Filter to Polymarket markets with condition_id, sorted by volume
        poly_markets = [m for m in markets if m.platform == "polymarket" and m.condition_id]
        poly_markets.sort(key=lambda m: m.volume_24h, reverse=True)
        target_markets = poly_markets[:top_n_markets]

        if not target_markets:
            log.debug("No Polymarket markets to discover wallets from")
            return 0

        # Step 1: Collect all unique holder addresses from target markets
        discovered_addresses: set[str] = set()
        for market in target_markets:
            try:
                holders = await self._data_client.get_top_holders(
                    market.condition_id or "", limit=20
                )
                for holder in holders:
                    address = holder.get("address", "")
                    if address:
                        discovered_addresses.add(address.lower())
                # Rate limit between API calls
                await asyncio.sleep(_API_DELAY_SECONDS)
            except Exception as e:
                log.warning(
                    "Failed to get holders for market",
                    market_id=market.external_id,
                    error=str(e),
                )

        if not discovered_addresses:
            log.debug("No holder addresses discovered")
            return 0

        log.info("Discovered holder addresses", count=len(discovered_addresses))

        # Step 2: Upsert all discovered wallets
        for address in discovered_addresses:
            try:
                await self._db.upsert_wallet(address, "polymarket")
            except Exception as e:
                log.warning("Failed to upsert wallet", address=address[:10], error=str(e))

        # Step 3: Filter to wallets needing score update
        addresses_needing_update = await self._db.get_wallets_needing_score_update(
            list(discovered_addresses),
            "polymarket",
            stale_hours=24,
        )

        log.info(
            "Wallets needing score update",
            total_discovered=len(discovered_addresses),
            needing_update=len(addresses_needing_update),
        )

        # Step 4: Score wallets and auto-watch high scorers
        newly_watched = 0
        scored_count = 0

        for address in addresses_needing_update:
            try:
                insider_score, total_trades = await self.update_wallet_metrics(address)
                scored_count += 1

                # Auto-watch if meets criteria
                if insider_score >= auto_watch_threshold and total_trades >= min_trades_to_watch:
                    await self._db.set_wallet_watched(address, "polymarket", True)
                    newly_watched += 1
                    log.info(
                        "Auto-watched wallet",
                        address=address[:10] + "...",
                        insider_score=f"{insider_score:.2f}",
                        total_trades=total_trades,
                    )

                # Rate limit between API calls
                await asyncio.sleep(_API_DELAY_SECONDS)
            except Exception as e:
                log.warning(
                    "Failed to score wallet",
                    address=address[:10],
                    error=str(e),
                )

        log.info(
            "Wallet discovery complete",
            discovered=len(discovered_addresses),
            scored=scored_count,
            newly_watched=newly_watched,
        )

        return newly_watched
