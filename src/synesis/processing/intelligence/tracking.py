"""Trade idea tracking — weekly review of open positions.

Looks up current prices for open trade ideas, updates price checkpoints
(1w, 2w, 1m), checks if target or stop was hit, and auto-expires ideas
older than 90 days.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from synesis.core.logging import get_logger

if TYPE_CHECKING:
    from synesis.providers.yfinance.client import YFinanceClient
    from synesis.storage.database import Database

logger = get_logger(__name__)

_EXPIRY_DAYS = 90


async def run_tracking_review(
    db: Database,
    yfinance: YFinanceClient,
) -> dict[str, Any]:
    """Weekly review of open trade ideas. Returns summary stats."""
    open_ideas = await db.get_open_trade_ideas()
    if not open_ideas:
        logger.info("No open trade ideas to review")
        return {"reviewed": 0, "closed": 0, "updated": 0}

    today = datetime.now(UTC).date()
    reviewed = 0
    closed = 0
    updated = 0
    skipped = 0

    for idea in open_ideas:
        ticker = idea["ticker"]
        idea_id = idea["id"]
        days_since = (today - idea["brief_date"]).days

        # Look up current price
        try:
            quote = await yfinance.get_quote(ticker)
            current_price = quote.last
            if current_price is None:
                logger.warning("No price for tracked ticker", ticker=ticker)
                skipped += 1
                continue
        except Exception:
            logger.exception("Failed to fetch quote for tracking", ticker=ticker)
            skipped += 1
            continue

        reviewed += 1
        updates: dict[str, Any] = {}

        # Update price checkpoints
        if days_since >= 7 and idea["price_at_1w"] is None:
            updates["price_at_1w"] = float(current_price)
        if days_since >= 14 and idea["price_at_2w"] is None:
            updates["price_at_2w"] = float(current_price)
        if days_since >= 30 and idea["price_at_1m"] is None:
            updates["price_at_1m"] = float(current_price)

        # Check target/stop hit
        entry = idea["entry_price"]
        target = idea["target_price"]
        stop = idea["stop_price"]
        direction = idea["direction"]

        if entry is not None and float(entry) > 0 and target is not None and stop is not None:
            if direction == "long":
                if float(current_price) >= float(target):
                    updates["status"] = "hit_target"
                    updates["pnl_at_close_pct"] = (
                        (float(current_price) - float(entry)) / float(entry) * 100
                    )
                    updates["closed_at"] = today
                    updates["close_reason"] = "Target hit"
                elif float(current_price) <= float(stop):
                    updates["status"] = "hit_stop"
                    updates["pnl_at_close_pct"] = (
                        (float(current_price) - float(entry)) / float(entry) * 100
                    )
                    updates["closed_at"] = today
                    updates["close_reason"] = "Stop hit"
            elif direction == "short":
                if float(current_price) <= float(target):
                    updates["status"] = "hit_target"
                    updates["pnl_at_close_pct"] = (
                        (float(entry) - float(current_price)) / float(entry) * 100
                    )
                    updates["closed_at"] = today
                    updates["close_reason"] = "Target hit"
                elif float(current_price) >= float(stop):
                    updates["status"] = "hit_stop"
                    updates["pnl_at_close_pct"] = (
                        (float(entry) - float(current_price)) / float(entry) * 100
                    )
                    updates["closed_at"] = today
                    updates["close_reason"] = "Stop hit"

        # Auto-expire after 90 days
        if "status" not in updates and days_since > _EXPIRY_DAYS:
            updates["status"] = "expired"
            if entry is not None and float(entry) > 0:
                updates["pnl_at_close_pct"] = (
                    (float(current_price) - float(entry)) / float(entry) * 100
                    if direction == "long"
                    else (float(entry) - float(current_price)) / float(entry) * 100
                )
            updates["closed_at"] = today
            updates["close_reason"] = f"Auto-expired after {_EXPIRY_DAYS} days"

        if updates:
            try:
                await db.update_trade_tracking(idea_id, updates)
                if "status" in updates:
                    closed += 1
                    logger.info(
                        "Trade idea closed",
                        ticker=ticker,
                        status=updates["status"],
                        pnl=updates.get("pnl_at_close_pct"),
                    )
                else:
                    updated += 1
            except Exception:
                logger.exception("Failed to update tracking", ticker=ticker, id=idea_id)

    if skipped:
        logger.warning("Tracking review skipped ideas due to price fetch failures", skipped=skipped)

    summary = {"reviewed": reviewed, "closed": closed, "updated": updated, "skipped": skipped}
    logger.info("Tracking review complete", **summary)
    return summary
