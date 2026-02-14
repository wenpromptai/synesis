"""Wallet intelligence tracker for Flow 3 (Polymarket only).

Tracks whale/insider wallets, monitors their trading activity,
calculates insider scores, and generates alerts when watched
wallets trade on markets we're scanning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from synesis.core.constants import (
    CONSISTENT_INSIDER_MIN_PNL_PER_POSITION,
    CONSISTENT_INSIDER_MIN_WIN_RATE,
    FAST_TRACK_MAX_WASH_RATIO,
    FRESH_INSIDER_MIN_POSITION,
    MARKET_INTEL_REDIS_PREFIX,
    WALLET_ACTIVITY_MAX_MARKETS,
    WALLET_API_DELAY_SECONDS,
    WALLET_DISCOVERY_TOP_N_MARKETS,
    WALLET_TOP_HOLDERS_LIMIT,
)
from synesis.core.logging import get_logger
from synesis.markets.models import HighConvictionTrade, InsiderAlert, UnifiedMarket

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from synesis.markets.polymarket import PolymarketDataClient
    from synesis.storage.database import Database

logger = get_logger(__name__)

_WALLETS_KEY = f"{MARKET_INTEL_REDIS_PREFIX}:wallets:watched"
_SCORE_PREFIX = f"{MARKET_INTEL_REDIS_PREFIX}:wallets:score"


@dataclass
class WalletMetricsResult:
    """Result of wallet scoring from positions + trades data."""

    insider_score: float
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_position_size: float
    focus_score: float
    freshness_score: float
    wash_trade_ratio: float
    position_count: int
    concentration: float  # max position / total portfolio
    largest_open_position: float  # largest currentValue among open positions (USDC)


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

    async def get_watched_wallets(self) -> list[dict[str, str | float | None]]:
        """Get watched wallets from DB (no caching).

        Returns:
            List of dicts with 'address', 'platform', 'insider_score',
            'specialty', 'watch_reason'
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
                    "specialty": w.get("specialty_category"),
                    "watch_reason": w.get("watch_reason"),
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
        watched_scores = {
            str(w["address"]).lower(): float(w.get("insider_score") or 0) for w in watched
        }
        watched_specialties = {str(w["address"]).lower(): w.get("specialty") for w in watched}
        watched_reasons = {str(w["address"]).lower(): w.get("watch_reason") for w in watched}

        alerts: list[InsiderAlert] = []

        # Only check Polymarket markets (Kalshi has no public wallet data)
        poly_markets = [m for m in markets if m.platform == "polymarket" and m.condition_id]

        for market in poly_markets[:WALLET_ACTIVITY_MAX_MARKETS]:
            try:
                holders = await self._data_client.get_top_holders(
                    market.condition_id or "",
                    limit=WALLET_TOP_HOLDERS_LIMIT,
                    yes_token_id=market.yes_token_id,
                )
                for holder in holders:
                    address = str(holder.get("address", "")).lower()
                    if address in watched_addresses:
                        score = watched_scores.get(address, 0)
                        if score >= self._insider_score_min:
                            # Determine trade direction from position
                            token_amount = abs(float(holder.get("amount", 0)))
                            outcome = holder.get("outcome", "")
                            # Convert tokens → USDC using current market price
                            price = (
                                market.yes_price
                                if outcome == "yes"
                                else market.no_price
                                if outcome == "no"
                                else market.yes_price  # fallback
                            )
                            trade_size_usd = token_amount * price

                            alerts.append(
                                InsiderAlert(
                                    market=market,
                                    wallet_address=address,
                                    insider_score=score,
                                    trade_direction=outcome or "unsure",
                                    trade_size=trade_size_usd,
                                    wallet_specialty=watched_specialties.get(address),
                                    watch_reason=watched_reasons.get(address),
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

    async def detect_high_conviction_trades(
        self,
        markets: list[UnifiedMarket],
        min_concentration: float = 0.50,
        max_positions: int = 5,
    ) -> list[HighConvictionTrade]:
        """Detect wallets making few, concentrated, large bets.

        A wallet with <=max_positions active positions where one position
        is >=min_concentration of total value = high conviction.

        Args:
            markets: Markets from the current scan
            min_concentration: Min fraction of portfolio in one market
            max_positions: Max active positions for a wallet to qualify

        Returns:
            List of high-conviction trade detections
        """
        if not self._data_client or not self._db:
            return []

        watched = await self.get_watched_wallets()
        if not watched:
            return []

        watched_by_addr = {
            str(w["address"]).lower(): w
            for w in watched
            if float(w.get("insider_score") or 0) >= self._insider_score_min
        }

        results: list[HighConvictionTrade] = []
        poly_markets = [m for m in markets if m.platform == "polymarket" and m.condition_id]

        # Collect (market, outcome) from holders — used only to identify which
        # watched wallets hold positions on which scanned markets.
        wallet_market_hits: dict[str, list[tuple[UnifiedMarket, str]]] = {}

        for market in poly_markets[:WALLET_ACTIVITY_MAX_MARKETS]:
            try:
                holders = await self._data_client.get_top_holders(
                    market.condition_id or "",
                    limit=WALLET_TOP_HOLDERS_LIMIT,
                    yes_token_id=market.yes_token_id,
                )
                for holder in holders:
                    address = str(holder.get("address", "")).lower()
                    if address in watched_by_addr:
                        outcome = holder.get("outcome", "")
                        wallet_market_hits.setdefault(address, []).append((market, outcome))
            except Exception as e:
                logger.warning(
                    "High-conviction holder check failed",
                    market_id=market.external_id,
                    error=str(e),
                )

        # For each wallet that appears, fetch their full positions (USDC values)
        checked_wallets: set[str] = set()
        for address, market_hits in wallet_market_hits.items():
            if address in checked_wallets:
                continue
            checked_wallets.add(address)

            try:
                all_positions, all_trades = await asyncio.gather(
                    self._data_client.get_wallet_positions(address),
                    self._data_client.get_wallet_trades(address, limit=500),
                )
                await asyncio.sleep(WALLET_API_DELAY_SECONDS)

                # Build conditionId → earliest trade timestamp
                earliest_trade: dict[str, datetime] = {}
                for trade in all_trades:
                    cid = trade.get("conditionId", "")
                    ts = trade.get("timestamp")
                    if cid and ts:
                        try:
                            dt = datetime.fromtimestamp(int(ts), tz=UTC)
                            if cid not in earliest_trade or dt < earliest_trade[cid]:
                                earliest_trade[cid] = dt
                        except (ValueError, OSError):
                            pass

                if not all_positions or len(all_positions) > max_positions:
                    continue

                # Build conditionId → position data from positions API
                # (currentValue is exact USDC, unlike holders amount which is tokens)
                pos_cv_by_condition: dict[str, float] = {}
                pos_iv_by_condition: dict[str, float] = {}
                pos_avgprice_by_condition: dict[str, float] = {}
                total_value = 0.0
                for pos in all_positions:
                    cv = abs(float(pos.get("currentValue", 0) or pos.get("initialValue", 0) or 0))
                    iv = abs(float(pos.get("initialValue", 0) or 0))
                    avg_price = float(pos.get("avgPrice", 0) or 0)
                    total_value += cv
                    cid = pos.get("conditionId", "")
                    if cid:
                        pos_cv_by_condition[cid] = pos_cv_by_condition.get(cid, 0) + cv
                        pos_iv_by_condition[cid] = pos_iv_by_condition.get(cid, 0) + iv
                        if avg_price > 0:
                            pos_avgprice_by_condition[cid] = avg_price

                if total_value <= 0:
                    continue

                # Check each market this wallet is in
                wallet_info = watched_by_addr[address]
                for market, outcome in market_hits:
                    cid = market.condition_id or ""
                    pos_size = pos_cv_by_condition.get(cid, 0)
                    if pos_size <= 0:
                        continue
                    concentration = pos_size / total_value
                    if concentration >= min_concentration:
                        results.append(
                            HighConvictionTrade(
                                market=market,
                                wallet_address=address,
                                insider_score=float(wallet_info.get("insider_score") or 0),
                                position_size=pos_size,
                                entry_cost=pos_iv_by_condition.get(cid, 0),
                                avg_entry_price=pos_avgprice_by_condition.get(cid, 0),
                                total_positions=len(all_positions),
                                concentration_pct=concentration,
                                trade_direction=outcome or "unsure",
                                wallet_specialty=wallet_info.get("specialty"),
                                entry_date=earliest_trade.get(cid),
                            )
                        )
            except Exception as e:
                logger.warning(
                    "High-conviction position check failed",
                    address=address[:10],
                    error=str(e),
                )

        if results:
            logger.info("High-conviction trades detected", count=len(results))

        return results

    async def discover_wallets_from_market(
        self, condition_id: str, yes_token_id: str | None = None
    ) -> list[str]:
        """Find new wallets worth watching from a market's top holders.

        Examines top holders and identifies wallets with profitable
        historical trading patterns. Newly discovered wallets are
        stored in the DB with is_watched=False (manual review needed).

        Args:
            condition_id: Polymarket condition ID
            yes_token_id: CLOB token ID for YES outcome (for holder direction)

        Returns:
            List of newly discovered wallet addresses
        """
        if not self._data_client or not self._db:
            return []

        discovered: list[str] = []
        try:
            holders = await self._data_client.get_top_holders(
                condition_id, limit=WALLET_TOP_HOLDERS_LIMIT, yes_token_id=yes_token_id
            )
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
                logger.debug(
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

    async def update_wallet_metrics(self, address: str) -> WalletMetricsResult | None:
        """Refresh wallet metrics from positions + trade history.

        Uses **/positions** for PnL and sizing (``cashPnl``, ``initialValue``,
        ``currentValue``) and **/trades** for activity and wash-trade detection.

        Sub-signals (weighted average):
        1. Profitability (0.30): position win_rate directly
        2. Freshness (0.15): Age < 30d → 1.0, < 90d → 0.5, else 0.0
        3. Focus (0.20): max(0, 1 - (position_count - 1) / 10)
        4. Sizing (0.20): min(1.0, avg_initialValue / 1000) — USDC
        5. Wash trade penalty (0.15): 1.0 - wash_ratio (from trades)

        Returns:
            ``WalletMetricsResult`` on success, ``None`` on failure or no data.
        """
        if not self._data_client or not self._db:
            return None

        try:
            # Fetch positions and trades concurrently
            positions, trades = await asyncio.gather(
                self._data_client.get_wallet_positions(address),
                self._data_client.get_wallet_trades(address, limit=500),
            )

            if not positions and not trades:
                return None

            # --- From POSITIONS: PnL, sizing, concentration ---
            # NOTE: cashPnl includes both realized and unrealized P&L.
            # We intentionally score on current portfolio state (conviction)
            # rather than historical trade outcomes, since the trades API
            # lacks P&L data.
            wins = 0
            total_pnl = 0.0
            initial_values: list[float] = []
            current_values: list[float] = []
            position_market_ids: set[str] = set()

            for pos in positions:
                cash_pnl = float(pos.get("cashPnl", 0) or 0)
                total_pnl += cash_pnl
                if cash_pnl > 0:
                    wins += 1

                init_val = float(pos.get("initialValue", 0) or 0)
                if init_val > 0:
                    initial_values.append(init_val)

                cur_val = float(pos.get("currentValue", 0) or 0)
                current_values.append(abs(cur_val))

                cond = pos.get("conditionId", "")
                if cond:
                    position_market_ids.add(cond)

            position_count = len(positions)
            position_win_rate = wins / position_count if position_count > 0 else 0.0
            avg_init_value = sum(initial_values) / len(initial_values) if initial_values else 0.0
            total_current = sum(current_values)
            max_current = max(current_values) if current_values else 0.0
            concentration = max_current / total_current if total_current > 0 else 0.0

            # Largest open position (currentValue > 0 = still active)
            open_values = [
                abs(float(pos.get("currentValue", 0) or 0))
                for pos in positions
                if float(pos.get("currentValue", 0) or 0) > 0
            ]
            largest_open = max(open_values) if open_values else 0.0

            # --- From TRADES: activity, wash detection ---
            total_trades = len(trades)
            trade_market_ids: set[str] = set()
            market_day_sides: dict[str, set[str]] = {}
            # Track trades per category for specialty
            category_trades: dict[str, list[bool]] = {}

            for trade in trades:
                market_id = trade.get("market", "") or trade.get("conditionId", "")
                if market_id:
                    trade_market_ids.add(market_id)

                side = trade.get("side", "")
                if market_id and side:
                    market_day_sides.setdefault(market_id, set()).add(side)

            unique_trade_markets = len(trade_market_ids)

            # Wash trade ratio: markets with both BUY and SELL / unique markets
            wash_count = sum(1 for sides in market_day_sides.values() if len(sides) > 1)
            wash_ratio = wash_count / max(unique_trade_markets, 1)

            # --- 5 Sub-signals ---
            # 1. Profitability (0.30): position win_rate directly
            profitability = position_win_rate

            # 2. Freshness (0.15)
            first_seen = await self._db.get_wallet_first_seen(address, "polymarket")
            freshness = 0.0
            if first_seen:
                age_days = (datetime.now(UTC) - first_seen).days
                if age_days < 30:
                    freshness = 1.0
                elif age_days < 90:
                    freshness = 0.5

            # 3. Focus (0.20): fewer positions = more focused
            focus = max(0.0, 1.0 - (position_count - 1) / 10) if position_count > 0 else 0.0
            focus = min(1.0, focus)

            # 4. Sizing (0.20): USDC-based, $1000 avg → 1.0
            sizing = min(1.0, avg_init_value / 1000.0) if avg_init_value > 0 else 0.0

            # 5. Wash trade penalty (0.15)
            wash_pen = max(0.0, 1.0 - wash_ratio)

            # Weighted average
            insider_score = min(
                1.0,
                profitability * 0.30
                + freshness * 0.15
                + focus * 0.20
                + sizing * 0.20
                + wash_pen * 0.15,
            )

            # Wallet specialty — best category by win rate
            all_market_ids = position_market_ids | trade_market_ids
            specialty_cat: str | None = None
            specialty_wr: float | None = None
            if all_market_ids and self._db:
                categories = await self._db.get_market_categories(list(all_market_ids))
                for pos in positions:
                    cond = pos.get("conditionId", "")
                    cat = categories.get(cond)
                    if cat:
                        is_win = float(pos.get("cashPnl", 0) or 0) > 0
                        category_trades.setdefault(cat, []).append(is_win)

                best_cat: str | None = None
                best_wr = 0.0
                for cat, results in category_trades.items():
                    if len(results) >= 2:
                        cat_wr = sum(results) / len(results)
                        if cat_wr > best_wr:
                            best_wr = cat_wr
                            best_cat = cat
                if best_cat:
                    specialty_cat = best_cat
                    specialty_wr = best_wr

            await self._db.upsert_wallet_metrics(
                address=address,
                platform="polymarket",
                total_trades=total_trades,
                wins=wins,
                win_rate=position_win_rate,
                total_pnl=total_pnl,
                insider_score=insider_score,
                unique_markets=len(all_market_ids),
                avg_position_size=avg_init_value,
                wash_trade_ratio=wash_ratio,
                profitability_score=profitability,
                focus_score=focus,
                sizing_score=sizing,
                freshness_score=freshness,
                wash_penalty=wash_pen,
                specialty_category=specialty_cat,
                specialty_win_rate=specialty_wr,
            )

            result = WalletMetricsResult(
                insider_score=insider_score,
                total_trades=total_trades,
                win_rate=position_win_rate,
                total_pnl=total_pnl,
                avg_position_size=avg_init_value,
                focus_score=focus,
                freshness_score=freshness,
                wash_trade_ratio=wash_ratio,
                position_count=position_count,
                concentration=concentration,
                largest_open_position=largest_open,
            )

            logger.debug(
                "Wallet metrics updated",
                address=address[:10] + "...",
                total_trades=total_trades,
                positions=position_count,
                win_rate=f"{position_win_rate:.2f}",
                insider_score=f"{insider_score:.2f}",
                specialty=specialty_cat,
            )
            return result
        except Exception as e:
            logger.error(
                "Wallet metrics update failed",
                address=address[:10] + "...",
                error=str(e),
            )
            return None

    async def re_score_watched_wallets(
        self,
        unwatch_threshold: float = 0.3,
        stale_hours: int = 24,
    ) -> int:
        """Re-score all watched wallets and demote those below threshold.

        Args:
            unwatch_threshold: Insider score below which wallets are demoted
            stale_hours: Only re-score wallets not scored within this many hours

        Returns:
            Number of demoted wallets
        """
        if not self._data_client or not self._db:
            return 0

        log = logger.bind(unwatch_threshold=unwatch_threshold, stale_hours=stale_hours)
        log.debug("Starting watched wallet re-score")

        addresses = await self._db.get_watched_wallets_needing_rescore(
            "polymarket", stale_hours=stale_hours
        )
        if not addresses:
            log.debug("No watched wallets need re-scoring")
            return 0

        log.debug("Watched wallets to re-score", count=len(addresses))

        demoted = 0
        for address in addresses:
            try:
                result = await self.update_wallet_metrics(address)

                if result is None:
                    log.warning("Wallet re-score returned no data, keeping watched", address=address[:10] + "...")
                elif result.insider_score < unwatch_threshold:
                    await self._db.set_wallet_watched(address, "polymarket", False)
                    demoted += 1
                    log.debug(
                        "Wallet demoted (unwatched)",
                        address=address[:10] + "...",
                        insider_score=f"{result.insider_score:.2f}",
                    )

                await asyncio.sleep(WALLET_API_DELAY_SECONDS)
            except Exception as e:
                log.warning(
                    "Failed to re-score wallet",
                    address=address[:10],
                    error=str(e),
                )

        log.debug(
            "Watched wallet re-score complete",
            total_rescored=len(addresses),
            demoted=demoted,
        )
        return demoted

    async def discover_and_score_wallets(
        self,
        markets: list[UnifiedMarket],
        top_n_markets: int = WALLET_DISCOVERY_TOP_N_MARKETS,
        auto_watch_threshold: float = 0.6,
        min_trades_to_watch: int = 20,
        unwatch_threshold: float = 0.3,
    ) -> int:
        """Discover wallets from top holders and auto-watch high scorers.

        This method:
        1. Takes the top N markets by volume
        2. Fetches top holders for each market (Polymarket only)
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
        log.debug("Starting wallet discovery pipeline")

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
                    market.condition_id or "",
                    limit=WALLET_TOP_HOLDERS_LIMIT,
                    yes_token_id=market.yes_token_id,
                )
                for holder in holders:
                    address = holder.get("address", "")
                    if address:
                        discovered_addresses.add(address.lower())
                # Rate limit between API calls
                await asyncio.sleep(WALLET_API_DELAY_SECONDS)
            except Exception as e:
                log.warning(
                    "Failed to get holders for market",
                    market_id=market.external_id,
                    error=str(e),
                )

        if not discovered_addresses:
            log.debug("No holder addresses discovered")
            return 0

        log.debug("Discovered holder addresses", count=len(discovered_addresses))

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

        log.debug(
            "Wallets needing score update",
            total_discovered=len(discovered_addresses),
            needing_update=len(addresses_needing_update),
        )

        # Step 4: Build set of currently watched addresses for demotion check
        watched_wallets = await self.get_watched_wallets()
        watched_addresses = {str(w["address"]).lower() for w in watched_wallets}

        # Step 5: Score wallets, auto-watch high scorers, demote degraded ones
        newly_watched = 0
        demoted = 0
        scored_count = 0

        for address in addresses_needing_update:
            try:
                result = await self.update_wallet_metrics(address)
                if result is None:
                    await asyncio.sleep(WALLET_API_DELAY_SECONDS)
                    continue
                scored_count += 1
                insider_score = result.insider_score
                total_trades = result.total_trades

                # Normal path: Auto-watch if meets score + trade-count thresholds
                if insider_score >= auto_watch_threshold and total_trades >= min_trades_to_watch:
                    await self._db.set_wallet_watched(
                        address, "polymarket", True, watch_reason="score"
                    )
                    newly_watched += 1
                    log.debug(
                        "Auto-watched wallet",
                        address=address[:10] + "...",
                        insider_score=f"{insider_score:.2f}",
                        total_trades=total_trades,
                        reason="score",
                    )
                # Fast-track: wallet below trade-count gate but qualitatively strong
                elif total_trades < min_trades_to_watch and self._qualifies_for_fast_track(result):
                    await self._db.set_wallet_watched(
                        address, "polymarket", True, watch_reason="high_conviction"
                    )
                    newly_watched += 1
                    log.debug(
                        "Fast-track auto-watched wallet",
                        address=address[:10] + "...",
                        insider_score=f"{insider_score:.2f}",
                        total_trades=total_trades,
                        reason="high_conviction",
                    )
                # Demotion: differentiated by watch_reason
                elif address in watched_addresses:
                    # Look up watch_reason for this wallet
                    wallet_info = next(
                        (w for w in watched_wallets if str(w["address"]).lower() == address),
                        None,
                    )
                    watch_reason = wallet_info.get("watch_reason") if wallet_info else None
                    threshold = 0.15 if watch_reason == "high_conviction" else unwatch_threshold
                    if insider_score < threshold:
                        await self._db.set_wallet_watched(address, "polymarket", False)
                        demoted += 1
                        log.debug(
                            "Wallet demoted during discovery",
                            address=address[:10] + "...",
                            insider_score=f"{insider_score:.2f}",
                            watch_reason=watch_reason,
                        )

                # Rate limit between API calls
                await asyncio.sleep(WALLET_API_DELAY_SECONDS)
            except Exception as e:
                log.warning(
                    "Failed to score wallet",
                    address=address[:10],
                    error=str(e),
                )

        log.debug(
            "Wallet discovery complete",
            discovered=len(discovered_addresses),
            scored=scored_count,
            newly_watched=newly_watched,
            demoted=demoted,
        )

        return newly_watched

    @staticmethod
    def _qualifies_for_fast_track(m: WalletMetricsResult) -> bool:
        """Check if a wallet qualifies for fast-track auto-watch.

        Hard filter: wash_trade_ratio must be below FAST_TRACK_MAX_WASH_RATIO.

        Path A — Consistent Insider: proven profitable big-money trader
          win_rate > CONSISTENT_INSIDER_MIN_WIN_RATE, PnL/position > CONSISTENT_INSIDER_MIN_PNL_PER_POSITION

        Path B — Fresh Insider: large focused open position (may have zero history)
          largest_open_position >= FRESH_INSIDER_MIN_POSITION
        """
        if m.wash_trade_ratio > FAST_TRACK_MAX_WASH_RATIO:
            return False

        # Path A: Consistent Insider
        if m.position_count > 0 and m.win_rate >= CONSISTENT_INSIDER_MIN_WIN_RATE:
            pnl_per_position = m.total_pnl / m.position_count
            if pnl_per_position >= CONSISTENT_INSIDER_MIN_PNL_PER_POSITION:
                return True

        # Path B: Fresh Insider — large open position, no prior history needed
        if m.largest_open_position >= FRESH_INSIDER_MIN_POSITION:
            return True

        return False
