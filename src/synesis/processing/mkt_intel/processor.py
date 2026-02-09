"""Flow 3 processor: orchestrates scanner + wallet tracker into signals.

Runs every 15 minutes to generate a MarketIntelSignal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from synesis.core.logging import get_logger
from synesis.markets.models import InsiderAlert, ScanResult, UnifiedMarket
from synesis.processing.mkt_intel.models import (
    MarketIntelOpportunity,
    MarketIntelSignal,
)
from synesis.processing.mkt_intel.scanner import MarketScanner
from synesis.processing.mkt_intel.wallets import WalletTracker

if TYPE_CHECKING:
    from synesis.config import Settings
    from synesis.markets.ws_manager import MarketWSManager
    from synesis.storage.database import Database

logger = get_logger(__name__)


class MarketIntelProcessor:
    """Orchestrates market scanner + wallet tracker into signals."""

    def __init__(
        self,
        settings: Settings,
        scanner: MarketScanner,
        wallet_tracker: WalletTracker,
        ws_manager: MarketWSManager | None = None,
        db: Database | None = None,
    ) -> None:
        self._settings = settings
        self._scanner = scanner
        self._wallet_tracker = wallet_tracker
        self._ws_manager = ws_manager
        self._db = db
        self._discovery_task: asyncio.Task[None] | None = None

    async def run_scan(self) -> MarketIntelSignal:
        """Run a full scan cycle and generate signal.

        1. scanner.scan() → trending + expiring + odds moves
        2. wallet_tracker.check_wallet_activity() → insider alerts
        3. Score opportunities
        4. Build and return signal
        """
        now = datetime.now(UTC)
        log = logger.bind(scan_time=now.isoformat())
        log.info("Market intel scan started")

        # 1. Run market scan
        scan_result = await self._scanner.scan()

        # 2. Check wallet activity on trending markets
        all_markets = list(
            {
                m.external_id: m
                for m in scan_result.trending_markets + scan_result.expiring_markets
            }.values()
        )
        insider_alerts = await self._wallet_tracker.check_wallet_activity(all_markets)

        # 3. Score opportunities
        opportunities = self._score_opportunities(
            scan_result=scan_result,
            insider_alerts=insider_alerts,
        )

        # 4. Calculate aggregate metrics
        uncertainty = self._calc_uncertainty_index(all_markets)
        informed_level = self._calc_informed_activity(insider_alerts)

        # 5. Build signal
        signal = MarketIntelSignal(
            timestamp=now,
            total_markets_scanned=scan_result.total_markets_scanned,
            trending_markets=scan_result.trending_markets[:20],
            expiring_soon=scan_result.expiring_markets[:20],
            insider_activity=insider_alerts[:10],
            odds_movements=scan_result.odds_movements[:10],
            opportunities=opportunities[:10],
            market_uncertainty_index=uncertainty,
            informed_activity_level=informed_level,
            ws_connected=self._ws_manager.is_connected if self._ws_manager else False,
        )

        # 6. Persist signal
        if self._db:
            try:
                await self._db.insert_mkt_intel_signal(signal)
            except Exception as e:
                log.error("Failed to persist mkt_intel signal", error=str(e))

        # 7. Discover and score new wallets (background task)
        if self._discovery_task and not self._discovery_task.done():
            log.debug("Previous wallet discovery still running, skipping")
        else:
            self._discovery_task = asyncio.create_task(
                self._run_wallet_discovery(scan_result.trending_markets),
                name="wallet_discovery",
            )

        log.info(
            "Market intel signal generated",
            markets_scanned=signal.total_markets_scanned,
            trending=len(signal.trending_markets),
            expiring=len(signal.expiring_soon),
            insider_alerts=len(signal.insider_activity),
            odds_movements=len(signal.odds_movements),
            opportunities=len(signal.opportunities),
        )

        return signal

    async def _run_wallet_discovery(self, trending_markets: list[UnifiedMarket]) -> None:
        """Run wallet discovery as a background task.

        Args:
            trending_markets: Markets to discover wallets from
        """
        try:
            await self._wallet_tracker.discover_and_score_wallets(
                trending_markets,
                top_n_markets=5,
                auto_watch_threshold=self._settings.mkt_intel_auto_watch_threshold,
            )
        except Exception as e:
            logger.error("Wallet discovery background task failed", error=str(e))

    def _score_opportunities(
        self,
        scan_result: ScanResult,
        insider_alerts: list[InsiderAlert],
    ) -> list[MarketIntelOpportunity]:
        """Combine triggers into scored opportunities."""

        # Build lookup maps
        odds_by_id: dict[str, float] = {}
        for om in scan_result.odds_movements:
            odds_by_id[om.market.external_id] = om.price_change_1h

        insider_by_id: dict[str, list[str]] = {}
        for ia in insider_alerts:
            insider_by_id.setdefault(ia.market.external_id, []).append(ia.wallet_address)

        expiring_ids = {m.external_id for m in scan_result.expiring_markets}

        # Score each market (dedup by external_id)
        market_map = {
            m.external_id: m for m in scan_result.trending_markets + scan_result.expiring_markets
        }

        # Compute volume threshold for high_volume trigger (80th percentile)
        all_volumes = sorted(m.volume_24h for m in market_map.values() if m.volume_24h > 0)
        if all_volumes:
            p80_idx = int(len(all_volumes) * 0.8)
            volume_threshold = all_volumes[min(p80_idx, len(all_volumes) - 1)]
        else:
            volume_threshold = float("inf")

        opportunities: list[MarketIntelOpportunity] = []

        for mid in market_map:
            market = market_map[mid]
            triggers: list[str] = []
            confidence = 0.0

            # Insider activity trigger
            insider_wallets = insider_by_id.get(mid, [])
            if insider_wallets:
                triggers.append("insider_activity")
                confidence += min(0.3, len(insider_wallets) * 0.15)

            # Odds movement trigger
            price_change = odds_by_id.get(mid)
            if price_change is not None:
                triggers.append("odds_movement")
                confidence += min(0.2, abs(price_change) * 2)

            # Expiring soon trigger — scaled by proximity
            hours_to_exp = None
            if mid in expiring_ids and market.end_date:
                hours_to_exp = (market.end_date - datetime.now(UTC)).total_seconds() / 3600
                if hours_to_exp > 0:
                    triggers.append("expiring_soon")
                    if hours_to_exp < 6:
                        confidence += 0.25
                    elif hours_to_exp < 12:
                        confidence += 0.15
                    else:
                        confidence += 0.10

            # High volume trigger — top 20% by 24h volume
            if market.volume_24h > 0 and market.volume_24h >= volume_threshold:
                triggers.append("high_volume")
                confidence += 0.15

            # Extreme odds trigger — near-certain outcomes
            if market.yes_price < 0.05 or market.yes_price > 0.95:
                triggers.append("extreme_odds")
                confidence += 0.10

            if not triggers:
                continue

            confidence = min(1.0, confidence)

            # Suggest direction based on available signals
            direction = "yes" if market.yes_price < 0.5 else "no"
            if price_change is not None:
                direction = "yes" if price_change > 0 else "no"

            reasoning_parts = []
            if insider_wallets:
                reasoning_parts.append(f"{len(insider_wallets)} insider wallet(s) active")
            if price_change is not None:
                reasoning_parts.append(f"Price moved {price_change:+.2f} in 1h")
            if hours_to_exp is not None:
                reasoning_parts.append(f"Expires in {hours_to_exp:.1f}h")
            if "high_volume" in triggers:
                reasoning_parts.append(f"High volume (${market.volume_24h:,.0f} 24h)")
            if "extreme_odds" in triggers:
                reasoning_parts.append(f"Extreme odds ({market.yes_price:.0%} YES)")

            opportunities.append(
                MarketIntelOpportunity(
                    market=market,
                    suggested_direction=direction,
                    confidence=confidence,
                    triggers=triggers,
                    insider_wallets_active=insider_wallets,
                    hours_to_expiration=hours_to_exp,
                    reasoning="; ".join(reasoning_parts),
                )
            )

        # Sort by confidence descending
        opportunities.sort(key=lambda o: o.confidence, reverse=True)
        return opportunities

    def _calc_uncertainty_index(self, markets: list[UnifiedMarket]) -> float:
        """Calculate market uncertainty index (avg distance from 0.5)."""
        if not markets:
            return 0.0
        # Markets closer to 0.5 = more uncertain
        total_uncertainty = sum(1.0 - abs(m.yes_price - 0.5) * 2 for m in markets)
        return total_uncertainty / len(markets)

    def _calc_informed_activity(self, insider_alerts: list[InsiderAlert]) -> float:
        """Calculate informed activity level (0-1 based on insider alerts)."""
        if not insider_alerts:
            return 0.0
        unique_wallets = len({a.wallet_address for a in insider_alerts})
        # Scale: 1 wallet = 0.2, 5+ wallets = 1.0
        return min(1.0, unique_wallets * 0.2)
