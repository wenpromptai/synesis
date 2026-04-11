"""MacroStrategist — assesses market regime from FRED data + Layer 1 themes.

Pre-fetches key economic indicators (VIX, yields, fed funds, unemployment),
formats them as structured context, then uses an LLM to interpret the regime,
produce sector tilts, and identify risks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.builtin_tools import WebSearchTool

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model, is_native_openai, native_search_docs
from synesis.processing.common.web_search import (
    Recency,
    format_search_results,
    read_web_page,
    search_market_impact,
)
from synesis.processing.intelligence.models import MacroView

if TYPE_CHECKING:
    from synesis.providers.fred.client import FREDClient

logger = get_logger(__name__)

_WEB_SEARCH_CAP = 3


_NATIVE_SEARCH_DESC = (
    "look up breaking economic events, policy changes, or verify FRED data context"
)

# Key FRED series for macro regime assessment
_FRED_SERIES = {
    "VIXCLS": "VIX (volatility index)",
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "FEDFUNDS": "Federal Funds Rate",
    "UNRATE": "Unemployment Rate",
}


@dataclass
class MacroStrategistDeps:
    """Dependencies for MacroStrategist."""

    fred: FREDClient
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    web_search_calls: int = 0


async def _fetch_fred_data(fred: FREDClient, lookback: int = 10) -> dict[str, Any]:
    """Fetch recent observations for key FRED series (latest + trend).

    Args:
        fred: FRED client.
        lookback: Number of recent observations to fetch per series.
    """
    data: dict[str, Any] = {}
    for series_id, label in _FRED_SERIES.items():
        try:
            obs = await fred.get_observations(series_id, limit=lookback, sort_order="desc")
            if obs and obs.observations:
                # Most recent first (desc order)
                history = [{"value": o.value, "date": str(o.date)} for o in obs.observations]
                latest = history[0]
                data[series_id] = {
                    "label": label,
                    "value": latest["value"],
                    "date": latest["date"],
                    "history": history,
                }
        except Exception:
            logger.warning("FRED fetch failed", series_id=series_id, exc_info=True)
            data[series_id] = {"label": label, "value": None, "date": None, "history": []}

    # Compute yield curve spread (latest)
    y10 = data.get("DGS10", {}).get("value")
    y2 = data.get("DGS2", {}).get("value")
    if y10 is not None and y2 is not None:
        try:
            data["yield_curve_spread"] = {
                "label": "Yield Curve Spread (10Y - 2Y)",
                "value": round(float(y10) - float(y2), 3),
            }
        except (ValueError, TypeError):
            logger.warning("Yield curve spread computation failed", y10=y10, y2=y2)

    return data


def _format_fred_context(fred_data: dict[str, Any]) -> str:
    """Format FRED data with full history for the LLM prompt."""
    lines = ["## Current Economic Indicators (FRED)"]
    for series_id, info in fred_data.items():
        val = info.get("value")
        label = info.get("label", series_id)
        if val is None:
            lines.append(f"- **{label}**: unavailable")
            continue

        # Show full history so LLM can see the shape of the trend
        history = info.get("history", [])
        if len(history) >= 2:
            # Reverse to chronological (oldest first) for readability
            points = " → ".join(
                f"{h['value']} ({h['date']})" for h in reversed(history) if h.get("value")
            )
            lines.append(f"- **{label}**: {points}")
        else:
            date_str = info.get("date", "")
            lines.append(f"- **{label}**: {val} (as of {date_str})")

    return "\n".join(lines)


def _format_macro_themes(state: dict[str, Any]) -> str:
    """Format Layer 1 macro themes for the LLM prompt."""
    lines = ["## Macro Themes from Layer 1 Analysts"]

    social = state.get("social_analysis", {})
    for theme in social.get("macro_themes", []):
        lines.append(f"- [Social] {theme.get('theme', '?')} — {theme.get('context', '')}")

    news = state.get("news_analysis", {})
    for theme in news.get("macro_themes", []):
        lines.append(f"- [News] {theme.get('theme', '?')} — {theme.get('context', '')}")

    # Include high-urgency macro news clusters
    for cluster in news.get("story_clusters", []):
        if cluster.get("event_type") == "macro":
            lines.append(
                f"- [News cluster] {cluster.get('headline', '?')} "
                f"(urgency: {cluster.get('urgency', '?')})"
            )

    if len(lines) == 1:
        lines.append("- No macro themes identified by Layer 1 analysts.")

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a senior macro strategist at a multi-strategy fund.
You assess the current market regime and produce sector tilts.

Today's date: {current_date}

## Your Job

1. **Regime Assessment**: Classify the current market as risk_on, risk_off, transitioning, or uncertain.
   - sentiment_score: -1.0 (strongly bearish) to 1.0 (strongly bullish) for the broad market.
   - Ground your assessment in the FRED data provided. Don't fabricate numbers.

2. **Key Drivers**: List 3-5 factors driving the current regime.

3. **Sector Tilts**: Which sectors/asset classes are favored or disfavored?
   - Can include equity sectors, commodities, fixed income, regions, themes.
   - sentiment_score per tilt: -1.0 (strongly underweight) to 1.0 (strongly overweight).
   - Sector tilts are tendencies, not guarantees. Note when current conditions deviate from
     typical regime patterns (e.g. risk-off doesn't always mean gold goes up).

4. **Risks**: What could shift the regime? List 2-4 scenarios.

## Tools

{native_search_docs}\
- `web_search(query, recency)` — Search via Brave API with explicit \
recency filter (day/week/month/year). Budget: {web_search_cap} calls.
- `web_read(url)` — Read a web page for full article content (~4000 chars). Unlimited calls.
  Use after web_search to read high-quality links that can strengthen your assessment.
- `get_fred_data(series_id)` — Fetch a FRED economic data series (last 5 observations).

## Rules
- Ground regime assessment in FRED data + Layer 1 themes, not speculation.
- Use web_search + web_read to get current context that FRED data lags on (e.g. breaking policy changes, geopolitical events).
- Sector tilts should be probabilistic, not prescriptive.
- sentiment_score calibration: ±0.8-1.0 = high conviction, ±0.4-0.7 = moderate, ±0.1-0.3 = weak.
"""


# ── Tool Functions ────────────────────────────────────────────────


async def _tool_web_search(
    ctx: RunContext[MacroStrategistDeps],
    query: str,
    recency: str = "day",
) -> str:
    """Search the web for macro context."""
    if ctx.deps.web_search_calls >= _WEB_SEARCH_CAP:
        return f"Web search budget exhausted ({_WEB_SEARCH_CAP} calls used)."
    ctx.deps.web_search_calls += 1

    valid_recencies = ("day", "week", "month", "year", "none")
    recency_val: Recency = recency if recency in valid_recencies else "day"  # type: ignore[assignment]
    results = await search_market_impact(query, recency=recency_val)
    if not results:
        return "No search results found."
    return format_search_results(results)


async def _tool_web_read(ctx: RunContext[MacroStrategistDeps], url: str) -> str:
    """Read a web page for full article content."""
    return await read_web_page(url)


async def _tool_get_fred_data(
    ctx: RunContext[MacroStrategistDeps],
    series_id: str,
) -> str:
    """Fetch a FRED economic data series."""
    try:
        obs = await ctx.deps.fred.get_observations(series_id, limit=5, sort_order="desc")
        if not obs or not obs.observations:
            return f"No data found for FRED series '{series_id}'."
        lines = [f"FRED {series_id} — last {len(obs.observations)} observations:"]
        for o in obs.observations:
            lines.append(f"  {o.date}: {o.value}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("FRED tool fetch failed", series_id=series_id, error=str(e), exc_info=True)
        return f"FRED fetch failed for '{series_id}': {e}"


# ── Public API ───────────────────────────────────────────────────


async def analyze_macro(
    state: dict[str, Any],
    deps: MacroStrategistDeps,
) -> MacroView:
    """Run the MacroStrategist on pipeline state."""
    logger.info("Starting MacroStrategist")

    # Pre-fetch FRED data
    fred_data = await _fetch_fred_data(deps.fred)
    fred_context = _format_fred_context(fred_data)
    themes_context = _format_macro_themes(state)

    # Build prompt
    user_prompt = f"{fred_context}\n\n{themes_context}"

    # Construct agent
    native = is_native_openai()
    agent: Agent[MacroStrategistDeps, MacroView] = Agent(
        model=create_model(tier="vsmart"),
        deps_type=MacroStrategistDeps,
        output_type=MacroView,
        system_prompt=SYSTEM_PROMPT.format(
            current_date=deps.current_date,
            web_search_cap=_WEB_SEARCH_CAP,
            native_search_docs=native_search_docs(_WEB_SEARCH_CAP, _NATIVE_SEARCH_DESC)
            if native
            else "",
        ),
        tools=[_tool_web_search, _tool_web_read, _tool_get_fred_data],
        builtin_tools=[WebSearchTool(max_uses=_WEB_SEARCH_CAP)] if native else [],
    )

    try:
        result = await agent.run(user_prompt, deps=deps)
        output: MacroView = result.output
    except Exception:
        logger.exception("MacroStrategist LLM call failed")
        return MacroView(
            regime="uncertain",
            sentiment_score=0.0,
            key_drivers=["[LLM synthesis failed]"],
            analysis_date=deps.current_date,
        )

    logger.info(
        "MacroStrategist complete",
        regime=output.regime,
        sentiment=output.sentiment_score,
        tilts=len(output.sector_tilts),
    )

    if output.analysis_date != deps.current_date:
        output = output.model_copy(update={"analysis_date": deps.current_date})

    return output
