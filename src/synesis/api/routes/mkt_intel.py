"""Market Intelligence (Flow 3) API endpoints."""

from __future__ import annotations

from typing import Any

import orjson
from fastapi import APIRouter, HTTPException

from synesis.core.constants import MARKET_INTEL_REDIS_PREFIX
from synesis.core.dependencies import AgentStateDep
from synesis.core.logging import get_logger
from synesis.storage.database import get_database

logger = get_logger(__name__)

router = APIRouter()

_LATEST_SIGNAL_QUERY = """
    SELECT time, payload
    FROM signals
    WHERE flow_id = 'mkt_intel'
    ORDER BY time DESC
    LIMIT 1
"""


@router.get("/latest")
async def get_latest_signal(state: AgentStateDep) -> dict[str, Any]:
    """Get the latest market intelligence signal."""
    try:
        db = get_database()
        row = await db.fetchrow(_LATEST_SIGNAL_QUERY)
        if row:
            return {
                "time": row["time"].isoformat(),
                "signal": orjson.loads(row["payload"]),
            }
        return {"time": None, "signal": None}
    except RuntimeError:
        return {"error": "Database not initialized"}


@router.post("/run")
async def trigger_manual_scan(state: AgentStateDep) -> dict[str, str]:
    """Trigger a manual market intelligence scan.

    The scan runs asynchronously; check /latest for results.
    """
    if state.scheduler and "mkt_intel" in state.trigger_fns:
        state.scheduler.add_job(
            state.trigger_fns["mkt_intel"],
            id="mkt_intel_manual",
            replace_existing=True,
        )
        return {"status": "scan_triggered"}
    raise HTTPException(status_code=503, detail="Market intelligence not enabled")


@router.get("/opportunities")
async def get_opportunities(state: AgentStateDep) -> dict[str, Any]:
    """Get current ranked opportunities from the latest signal."""
    try:
        db = get_database()
        row = await db.fetchrow(_LATEST_SIGNAL_QUERY)
        if row:
            payload = orjson.loads(row["payload"])
            opportunities = payload.get("opportunities", [])
            return {
                "opportunities": opportunities,
                "count": len(opportunities),
            }
        return {"opportunities": [], "count": 0}
    except RuntimeError:
        return {"error": "Database not initialized"}


@router.get("/wallets")
async def get_watched_wallets(state: AgentStateDep) -> dict[str, Any]:
    """Get watched wallets with metrics."""
    try:
        db = get_database()
        rows = await db.get_watched_wallets("polymarket")
        wallets = [
            {
                "address": row["address"],
                "platform": row["platform"],
                "insider_score": float(row["insider_score"] or 0),
                "win_rate": float(row["win_rate"] or 0),
                "total_trades": int(row["total_trades"] or 0),
            }
            for row in rows
        ]
        return {"wallets": wallets, "count": len(wallets)}
    except RuntimeError:
        return {"error": "Database not initialized"}


@router.get("/ws-status")
async def get_ws_status(state: AgentStateDep) -> dict[str, Any]:
    """Get WebSocket connection health status and live volume accumulation."""
    poly_connected = False
    kalshi_connected = False
    total_subscribed = 0

    try:
        # Check if WS health info is stored in Redis
        health = await state.redis.hgetall(f"{MARKET_INTEL_REDIS_PREFIX}:ws:health")  # type: ignore[misc]
        if health:
            poly_connected = health.get(b"poly_connected", b"0") == b"1"
            kalshi_connected = health.get(b"kalshi_connected", b"0") == b"1"
            total_subscribed = int(health.get(b"total_subscribed", b"0"))
    except Exception as e:
        logger.error("WS health check failed", error=str(e))

    # Read accumulated volume for subscribed markets (top 20)
    volume_prefix = f"{MARKET_INTEL_REDIS_PREFIX}:ws:volume_1h"
    live_volumes: dict[str, float] = {}
    try:
        for platform in ("polymarket", "kalshi"):
            cursor = 0
            while True:
                cursor, keys = await state.redis.scan(
                    cursor=cursor,
                    match=f"{volume_prefix}:{platform}:*",
                    count=100,
                )
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    val = await state.redis.get(key_str)
                    if val:
                        try:
                            vol = float(val)
                            if vol > 0:
                                market_id = key_str.split(":")[-1]
                                live_volumes[f"{platform}:{market_id}"] = vol
                        except (ValueError, TypeError):
                            pass
                if cursor == 0:
                    break
    except Exception as e:
        logger.error("Failed to read live volumes", error=str(e))

    # Sort by volume descending, take top 20
    sorted_volumes = dict(sorted(live_volumes.items(), key=lambda x: x[1], reverse=True)[:20])

    return {
        "polymarket_ws": poly_connected,
        "kalshi_ws": kalshi_connected,
        "total_subscribed_markets": total_subscribed,
        "live_volumes": sorted_volumes,
    }


@router.post("/wallets/discover")
async def trigger_wallet_discovery(state: AgentStateDep) -> dict[str, Any]:
    """Manually trigger wallet discovery on current trending markets.

    Fetches top holders from trending Polymarket markets, scores them,
    and auto-watches wallets with high insider scores.

    Returns:
        Status and count of newly watched wallets
    """
    from synesis.config import get_settings
    from synesis.markets.polymarket import PolymarketClient, PolymarketDataClient
    from synesis.processing.mkt_intel.scanner import _poly_to_unified
    from synesis.processing.mkt_intel.wallets import WalletTracker

    settings = get_settings()

    try:
        db = get_database()
    except RuntimeError:
        return {"status": "error", "detail": "Database not initialized"}

    gamma_client = PolymarketClient()
    data_client = PolymarketDataClient()

    try:
        trending = await gamma_client.get_trending_markets(limit=20)

        if not trending:
            return {"status": "no_markets", "newly_watched": 0}

        markets = [_poly_to_unified(m) for m in trending]

        tracker = WalletTracker(
            redis=state.redis,
            db=db,
            data_client=data_client,
            insider_score_min=settings.mkt_intel_insider_score_min,
        )

        newly_watched = await tracker.discover_and_score_wallets(
            markets,
            top_n_markets=5,
            auto_watch_threshold=settings.mkt_intel_auto_watch_threshold,
        )

        return {
            "status": "completed",
            "markets_scanned": len(markets),
            "newly_watched": newly_watched,
        }
    except Exception as e:
        logger.error("Wallet discovery endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail="Wallet discovery failed") from e
    finally:
        await gamma_client.close()
        await data_client.close()
