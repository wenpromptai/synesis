"""PriceAnalyst — technical + options analysis per ticker.

Three-phase pipeline:
1. Data gathering: yfinance (bars, quote, realized vol) + Massive (ATM options EOD bars)
2. Deterministic computation: pandas-ta indicators + IV from options prices + pattern flags
3. LLM synthesis: interpret indicators and options landscape in plain English

yfinance: outrights only (bars, quote, options snapshot for realized vol)
Massive: options only (contract lookup, ATM call/put EOD bars)
IV: self-computed from Massive EOD close prices via Newton-Raphson BS inversion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai import Agent

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.intelligence.models import PriceAnalysis
from synesis.processing.intelligence.specialists.price.indicators import (
    bars_to_dataframe,
    compute_indicators,
    compute_iv_from_price,
    detect_notable_setups,
)

if TYPE_CHECKING:
    from synesis.providers.massive.client import MassiveClient
    from synesis.providers.yfinance.client import YFinanceClient

logger = get_logger(__name__)


@dataclass
class PriceDeps:
    """Dependencies for PriceAnalyst."""

    yfinance: YFinanceClient
    massive: MassiveClient | None = None
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())


# ── Phase 1: Data Gathering ──────────────────────────────────────


async def _gather_yfinance(yf: YFinanceClient, ticker: str) -> dict[str, Any]:
    """Fetch quote + 3mo bars from yfinance (outrights only, no options)."""
    quote = await yf.get_quote(ticker)
    bars = await yf.get_history(ticker, period="3mo", interval="1d")

    # Realized vol from options snapshot (the one yfinance metric that's reliable)
    realized_vol = None
    try:
        snapshot = await yf.get_options_snapshot(ticker)
        if snapshot and snapshot.realized_vol_30d is not None:
            realized_vol = round(snapshot.realized_vol_30d * 100, 1)
    except Exception:
        logger.warning("Options snapshot failed — realized_vol unavailable", ticker=ticker)

    return {"quote": quote, "bars": bars, "realized_vol_30d": realized_vol}


async def _gather_massive(
    massive: MassiveClient, ticker: str, spot: float | None
) -> dict[str, Any]:
    """Fetch ATM options data from Massive (3 calls per ticker).

    Calls (3 per ticker):
    1. get_options_contracts — find ATM call + put tickers for nearest monthly DTE > 20
    2. get_bars(atm_call) — call EOD price history
    3. get_bars(atm_put) — put EOD price history

    Short interest data is available via CompanyAnalyst (yfinance fundamentals).
    """
    data: dict[str, Any] = {
        "atm_call_close": None,
        "atm_put_close": None,
        "atm_call_volume": None,
        "atm_put_volume": None,
        "strike": None,
        "expiration": None,
        "dte": None,
    }

    if not spot or spot <= 0:
        return data

    today = datetime.now(UTC).date()

    # Call 1: find ATM contracts for nearest monthly DTE > 20
    try:
        atm_strike = round(spot / 5) * 5  # round to nearest $5
        # Search a range of expirations and pick the first with DTE > 20
        exp_start = (today + timedelta(days=21)).isoformat()
        exp_end = (today + timedelta(days=60)).isoformat()

        contracts = await massive.get_options_contracts(
            ticker,
            expiration_date_gte=exp_start,
            expiration_date_lte=exp_end,
            strike_price=atm_strike,
            limit=10,
        )

        if not contracts:
            # Try without exact strike, broader range
            contracts = await massive.get_options_contracts(
                ticker,
                expiration_date_gte=exp_start,
                expiration_date_lte=exp_end,
                strike_price_gte=atm_strike - 5,
                strike_price_lte=atm_strike + 5,
                limit=10,
            )

        # Separate calls and puts
        call_contract = next((c for c in contracts if c.contract_type == "call"), None)
        put_contract = next(
            (
                c
                for c in contracts
                if c.contract_type == "put"
                and c.strike_price == (call_contract.strike_price if call_contract else 0)
            ),
            None,
        )

        if call_contract:
            data["strike"] = call_contract.strike_price
            data["expiration"] = call_contract.expiration_date
            dte = (date.fromisoformat(call_contract.expiration_date) - today).days
            data["dte"] = dte

            # Call 2: ATM call EOD bars
            from_date = (today - timedelta(days=14)).isoformat()
            to_date = today.isoformat()
            call_bars = await massive.get_bars(call_contract.ticker, 1, "day", from_date, to_date)
            if call_bars and call_bars.bars:
                latest = call_bars.bars[-1]
                data["atm_call_close"] = latest.close
                data["atm_call_volume"] = int(latest.volume)

            # Call 3: ATM put EOD bars
            if put_contract:
                put_bars = await massive.get_bars(put_contract.ticker, 1, "day", from_date, to_date)
                if put_bars and put_bars.bars:
                    latest = put_bars.bars[-1]
                    data["atm_put_close"] = latest.close
                    data["atm_put_volume"] = int(latest.volume)

    except Exception:
        logger.warning("Massive options data failed", ticker=ticker, exc_info=True)

    return data


# ── Phase 2: Deterministic Computation ───────────────────────────


def _derive_options_metrics(
    massive_data: dict[str, Any],
    spot: float | None,
    realized_vol: float | None,
) -> dict[str, Any]:
    """Derive options metrics from Massive EOD close prices + self-computed IV."""
    metrics: dict[str, Any] = {}

    if realized_vol is not None:
        metrics["realized_vol_30d"] = realized_vol

    call_close = massive_data.get("atm_call_close")
    put_close = massive_data.get("atm_put_close")
    strike = massive_data.get("strike")
    dte = massive_data.get("dte")

    if not spot or not strike or not dte or dte <= 0:
        return metrics

    metrics["days_to_expiry"] = dte
    tte = dte / 365.0

    # Compute IV from Massive EOD close prices
    call_iv = None
    put_iv = None

    if call_close and call_close > 0:
        call_iv = compute_iv_from_price(call_close, spot, strike, tte, option_type="call")
    if put_close and put_close > 0:
        put_iv = compute_iv_from_price(put_close, spot, strike, tte, option_type="put")

    # ATM IV (average of call + put)
    ivs = [iv for iv in [call_iv, put_iv] if iv is not None]
    if ivs:
        atm_iv_pct = sum(ivs) / len(ivs) * 100
        metrics["atm_iv"] = round(atm_iv_pct, 1)

        # IV-RV spread
        if realized_vol is not None:
            metrics["iv_rv_spread"] = round(atm_iv_pct - realized_vol, 1)

    # ATM skew (put IV / call IV)
    if call_iv and put_iv and call_iv > 0:
        metrics["atm_skew_ratio"] = round(put_iv / call_iv, 2)

    # Put/call volume ratio from Massive bars
    call_vol = massive_data.get("atm_call_volume")
    put_vol = massive_data.get("atm_put_volume")
    if call_vol and call_vol > 0 and put_vol is not None:
        metrics["put_call_volume_ratio"] = round(put_vol / call_vol, 2)

    return metrics


# ── Phase 3: LLM Synthesis ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert technical analyst and options strategist.

Today's date: {current_date}

## Your Job

You receive pre-computed technical indicators and options metrics for a specific ticker.
Produce two narratives:

1. **technical_narrative**: 3-5 sentences on the price/technical picture.
   - Synthesize trend (EMA cross, ADX), momentum (RSI, MACD), volatility (BB width, ATR%),
     volume (OBV, volume ratio), key levels (support/resistance), and mean reversion (z-score).
   - Highlight what's notable: divergences, extremes, squeezes, level tests.
   - Cite the actual values — "RSI at 54" not "momentum is okay."

2. **options_narrative**: 3-5 sentences on what the options market reveals.
   - IV computed from EOD options close prices. Compare ATM IV to 30-day realized vol.
   - Note the skew (put IV / call IV) and put/call volume ratio.
   - If any metrics are null, note what's available and interpret that.

Cite specific numbers. Focus on what the data shows, not what to trade.
"""


class _PriceNarratives(BaseModel):
    """LLM output — just the two narratives, not the full PriceAnalysis."""

    technical_narrative: str
    options_narrative: str


# ── Public API ───────────────────────────────────────────────────


async def analyze_price(ticker: str, deps: PriceDeps) -> PriceAnalysis:
    """Run the PriceAnalyst three-phase pipeline for a single ticker."""
    ticker = ticker.upper()
    logger.info("Starting PriceAnalyst", ticker=ticker)

    # Phase 1: Gather data
    yf_data = await _gather_yfinance(deps.yfinance, ticker)
    quote = yf_data["quote"]
    spot = quote.last
    realized_vol = yf_data.get("realized_vol_30d")

    massive_data: dict[str, Any] = {}
    if deps.massive:
        massive_data = await _gather_massive(deps.massive, ticker, spot)

    # Phase 2: Compute indicators
    df = bars_to_dataframe(yf_data["bars"])
    indicators = compute_indicators(df)
    options_metrics = _derive_options_metrics(massive_data, spot, realized_vol)

    # Detect patterns
    notable = detect_notable_setups(indicators, options_metrics)

    # Build partial analysis (deterministic fields)
    change_1d = None
    if quote.last and quote.prev_close and quote.prev_close > 0:
        change_1d = round((quote.last - quote.prev_close) / quote.prev_close * 100, 2)

    partial = PriceAnalysis(
        ticker=ticker,
        analysis_date=deps.current_date,
        spot_price=spot,
        change_1d_pct=change_1d,
        **{k: v for k, v in indicators.items() if v is not None},
        **{k: v for k, v in options_metrics.items() if v is not None},
        notable_setups=notable,
    )

    logger.info(
        "Phase 2 complete",
        ticker=ticker,
        rsi=indicators.get("rsi_14"),
        atm_iv=options_metrics.get("atm_iv"),
        setups=len(notable),
    )

    # Phase 3: LLM synthesis — only produces narratives
    agent: Agent[None, _PriceNarratives] = Agent(
        model=create_model(tier="vsmart"),
        output_type=_PriceNarratives,
        system_prompt=SYSTEM_PROMPT.format(current_date=deps.current_date),
    )

    user_prompt = f"# Price Analysis: {ticker}\n\n"
    user_prompt += f"Spot: ${spot:.2f}\n" if spot else ""
    user_prompt += f"Change: {change_1d:+.1f}%\n" if change_1d else ""
    user_prompt += f"\n## Technical Indicators\n{partial.model_dump_json(indent=2)}\n"
    if notable:
        user_prompt += "\n## Notable Setups\n" + "\n".join(f"- {s}" for s in notable) + "\n"

    try:
        result = await agent.run(user_prompt)
        partial = partial.model_copy(
            update={
                "technical_narrative": result.output.technical_narrative,
                "options_narrative": result.output.options_narrative,
            }
        )
    except Exception:
        logger.exception("PriceAnalyst LLM synthesis failed", ticker=ticker)
        partial = partial.model_copy(
            update={
                "technical_narrative": "[LLM synthesis failed]",
                "options_narrative": "[LLM synthesis failed]",
            }
        )

    logger.info("PriceAnalyst complete", ticker=ticker)
    return partial
