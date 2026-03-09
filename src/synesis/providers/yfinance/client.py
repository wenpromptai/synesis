"""yfinance client — async wrapper around the synchronous yfinance library.

Free data, no API key. ~15 min delayed for US equities during market hours.
All yfinance calls are wrapped in asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, date as date_cls, datetime
from typing import TYPE_CHECKING, Any

import orjson
import yfinance as yf

from synesis.config import get_settings
from synesis.core.logging import get_logger
from synesis.core.constants import (
    OPTIONS_SNAPSHOT_ATM_STRIKES,
    OPTIONS_SNAPSHOT_MIN_BARS,
    OPTIONS_SNAPSHOT_MIN_DTE,
    OPTIONS_SNAPSHOT_TRADING_DAYS,
)
from synesis.providers.yfinance.models import (
    EquityQuote,
    FXRate,
    OHLCBar,
    OptionsChain,
    OptionsContract,
    OptionsGreeks,
    OptionsSnapshot,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

CACHE_PREFIX = "synesis:yfinance"


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None for NaN/Inf/missing."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    """Convert a value to int, returning None for NaN/missing."""
    f = _safe_float(val)
    if f is None:
        return None
    return int(f)


def _compute_bs_greeks(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    iv: float,
) -> OptionsGreeks:
    """Compute Black-Scholes Greeks from implied volatility."""
    from scipy.stats import norm

    if time_to_expiry <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return OptionsGreeks(implied_volatility=iv if iv > 0 else None)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * iv**2) * time_to_expiry) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    exp_rt = math.exp(-risk_free_rate * time_to_expiry)
    pdf_d1: float = norm.pdf(d1)

    if option_type == "call":
        delta: float = norm.cdf(d1)
        rho = strike * time_to_expiry * exp_rt * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        rho = -strike * time_to_expiry * exp_rt * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (spot * iv * sqrt_t)
    theta = (
        -(spot * pdf_d1 * iv) / (2 * sqrt_t)
        - risk_free_rate
        * strike
        * exp_rt
        * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))
    ) / 365
    vega = spot * pdf_d1 * sqrt_t / 100

    return OptionsGreeks(
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        rho=round(rho, 6),
        implied_volatility=round(iv, 6),
    )


class YFinanceClient:
    """Async client for yfinance data with Redis caching.

    Usage:
        client = YFinanceClient(redis=redis_client)
        quote = await client.get_quote("AAPL")
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def get_quote(self, ticker: str) -> EquityQuote:
        """Get a snapshot quote for an equity/ETF/index."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:quote:{ticker.upper()}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return EquityQuote.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        info = await asyncio.to_thread(_fetch_quote_info, ticker)

        quote = EquityQuote(
            ticker=ticker.upper(),
            name=info.get("shortName") or info.get("longName"),
            currency=info.get("currency"),
            exchange=info.get("exchange"),
            last=_safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
            prev_close=_safe_float(
                info.get("previousClose") or info.get("regularMarketPreviousClose")
            ),
            open=_safe_float(info.get("open") or info.get("regularMarketOpen")),
            high=_safe_float(info.get("dayHigh") or info.get("regularMarketDayHigh")),
            low=_safe_float(info.get("dayLow") or info.get("regularMarketDayLow")),
            volume=_safe_int(info.get("volume") or info.get("regularMarketVolume")),
            market_cap=_safe_float(info.get("marketCap")),
            avg_50d=_safe_float(info.get("fiftyDayAverage")),
            avg_200d=_safe_float(info.get("twoHundredDayAverage")),
        )

        await self._redis.set(
            cache_key,
            orjson.dumps(quote.model_dump(mode="json")),
            ex=settings.yfinance_cache_ttl_quote,
        )
        return quote

    async def get_history(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> list[OHLCBar]:
        """Get OHLCV history bars."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:history:{ticker.upper()}:{period}:{interval}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return [OHLCBar.model_validate(b) for b in orjson.loads(cached)]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        bars = await asyncio.to_thread(_fetch_history, ticker, period, interval)

        await self._redis.set(
            cache_key,
            orjson.dumps([b.model_dump(mode="json") for b in bars]),
            ex=settings.yfinance_cache_ttl_history,
        )
        return bars

    async def get_fx_rate(self, pair: str) -> FXRate:
        """Get FX spot rate. pair should be like 'EURUSD=X'."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:fx:{pair.upper()}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return FXRate.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        info = await asyncio.to_thread(_fetch_quote_info, pair)

        rate = FXRate(
            pair=pair.upper(),
            rate=_safe_float(info.get("regularMarketPrice")),
            bid=_safe_float(info.get("bid")),
            ask=_safe_float(info.get("ask")),
        )

        await self._redis.set(
            cache_key,
            orjson.dumps(rate.model_dump(mode="json")),
            ex=settings.yfinance_cache_ttl_fx,
        )
        return rate

    async def get_options_expirations(self, ticker: str) -> list[str]:
        """Get available options expiration dates."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:opt_exp:{ticker.upper()}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return orjson.loads(cached)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        expirations = await asyncio.to_thread(_fetch_options_expirations, ticker)

        await self._redis.set(
            cache_key,
            orjson.dumps(expirations),
            ex=settings.yfinance_cache_ttl_options,
        )
        return expirations

    async def get_options_chain(
        self,
        ticker: str,
        expiration: str,
        greeks: bool = False,
    ) -> OptionsChain:
        """Get full options chain for a given expiration date."""
        settings = get_settings()
        cache_key = f"{CACHE_PREFIX}:chain:{ticker.upper()}:{expiration}:g{int(greeks)}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return OptionsChain.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        chain = await asyncio.to_thread(_fetch_options_chain, ticker, expiration, greeks)

        await self._redis.set(
            cache_key,
            orjson.dumps(chain.model_dump(mode="json")),
            ex=settings.yfinance_cache_ttl_options,
        )
        return chain

    async def get_options_snapshot(
        self,
        ticker: str,
        greeks: bool = True,
    ) -> OptionsSnapshot:
        """Get pre-computed options snapshot with realized vol and ATM chain.

        Orchestrates existing methods: get_quote, get_history,
        get_options_expirations, get_options_chain. Computes 30d realized vol
        and filters chain to nearest-ATM strikes.
        """
        settings = get_settings()
        ticker_up = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:snapshot:{ticker_up}:g{int(greeks)}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return OptionsSnapshot.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        # Spot price
        quote = await self.get_quote(ticker_up)
        spot = quote.last

        # 30d realized vol from daily closes
        realized_vol: float | None = None
        bars = await self.get_history(ticker_up, period="1mo", interval="1d")
        if len(bars) >= OPTIONS_SNAPSHOT_MIN_BARS:
            closes = [b.close for b in bars if b.close is not None and b.close > 0]
            if len(closes) >= OPTIONS_SNAPSHOT_MIN_BARS:
                log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                mean_r = sum(log_returns) / len(log_returns)
                variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
                realized_vol = math.sqrt(variance * OPTIONS_SNAPSHOT_TRADING_DAYS)

        # Select expiry: skip < MIN_DTE days, pick first valid
        expirations = await self.get_options_expirations(ticker_up)
        today = datetime.now(UTC).date()
        target_exp = ""
        days_to_expiry = 0
        for exp_str in expirations:
            exp_date = date_cls.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte >= OPTIONS_SNAPSHOT_MIN_DTE:
                target_exp = exp_str
                days_to_expiry = dte
                break

        if not target_exp:
            return OptionsSnapshot(
                ticker=ticker_up,
                spot=spot,
                realized_vol_30d=realized_vol,
                expiration="",
                days_to_expiry=0,
                calls=[],
                puts=[],
            )

        chain = await self.get_options_chain(ticker_up, target_exp, greeks=greeks)

        # Filter to N nearest-ATM strikes per side (skip if no spot price)
        if spot is not None:
            calls = sorted(chain.calls, key=lambda c: abs(c.strike - spot))[
                :OPTIONS_SNAPSHOT_ATM_STRIKES
            ]
            puts = sorted(chain.puts, key=lambda c: abs(c.strike - spot))[
                :OPTIONS_SNAPSHOT_ATM_STRIKES
            ]
        else:
            calls = chain.calls[:OPTIONS_SNAPSHOT_ATM_STRIKES]
            puts = chain.puts[:OPTIONS_SNAPSHOT_ATM_STRIKES]

        snapshot = OptionsSnapshot(
            ticker=ticker_up,
            spot=spot,
            realized_vol_30d=round(realized_vol, 6) if realized_vol is not None else None,
            expiration=target_exp,
            days_to_expiry=days_to_expiry,
            calls=calls,
            puts=puts,
        )

        await self._redis.set(
            cache_key,
            orjson.dumps(snapshot.model_dump(mode="json")),
            ex=settings.yfinance_cache_ttl_options,
        )
        return snapshot


# ---------------------------------------------------------------------------
# Synchronous helpers (run inside asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_quote_info(ticker: str) -> dict[str, Any]:
    """Fetch ticker .info dict (synchronous)."""
    try:
        t = yf.Ticker(ticker)
        return dict(t.info or {})
    except Exception as e:
        logger.warning("yfinance info fetch failed", ticker=ticker, error=str(e))
        return {}


def _fetch_history(ticker: str, period: str, interval: str) -> list[OHLCBar]:
    """Fetch OHLCV history (synchronous)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df is None or df.empty:
            return []

        bars: list[OHLCBar] = []
        for idx, row in df.iterrows():
            bars.append(
                OHLCBar(
                    date=idx.date() if hasattr(idx, "date") else idx,
                    open=_safe_float(row.get("Open")),
                    high=_safe_float(row.get("High")),
                    low=_safe_float(row.get("Low")),
                    close=_safe_float(row.get("Close")),
                    volume=_safe_int(row.get("Volume")),
                )
            )
        return bars
    except Exception as e:
        logger.warning("yfinance history fetch failed", ticker=ticker, error=str(e))
        return []


def _fetch_options_expirations(ticker: str) -> list[str]:
    """Fetch available option expiration dates (synchronous)."""
    try:
        t = yf.Ticker(ticker)
        return list(t.options)
    except Exception as e:
        logger.warning("yfinance options expirations failed", ticker=ticker, error=str(e))
        return []


def _fetch_options_chain(ticker: str, expiration: str, compute_greeks: bool) -> OptionsChain:
    """Fetch full options chain for an expiration (synchronous)."""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration)
    except Exception as e:
        logger.warning("yfinance option chain failed", ticker=ticker, error=str(e))
        return OptionsChain(ticker=ticker.upper(), expiration=expiration, calls=[], puts=[])

    spot = _safe_float((t.info or {}).get("regularMarketPrice")) or 0.0

    # Time to expiry in years
    exp_date = date_cls.fromisoformat(expiration)
    today = date_cls.today()
    tte = max((exp_date - today).days, 0) / 365.0
    risk_free = 0.05  # Approximate 10Y treasury

    def _parse_contracts(
        df: Any,
        option_type: str,  # noqa: ANN401
    ) -> list[OptionsContract]:
        contracts: list[OptionsContract] = []
        if df is None or (hasattr(df, "empty") and df.empty):
            return contracts
        for _, row in df.iterrows():
            # Skip contracts with stale data — only include if last trade was today
            last_trade = row.get("lastTradeDate")
            if last_trade is not None:
                try:
                    trade_date = getattr(last_trade, "date", lambda: None)()
                    if trade_date != today:
                        continue
                except Exception:
                    continue

            iv = _safe_float(row.get("impliedVolatility"))
            strike = _safe_float(row.get("strike"))

            greeks_obj: OptionsGreeks | None = None
            if compute_greeks and iv and strike and spot > 0:
                greeks_obj = _compute_bs_greeks(
                    option_type=option_type,
                    spot=spot,
                    strike=strike,
                    time_to_expiry=tte,
                    risk_free_rate=risk_free,
                    iv=iv,
                )

            contracts.append(
                OptionsContract(
                    contract_symbol=str(row.get("contractSymbol", "")),
                    strike=strike or 0.0,
                    last_price=_safe_float(row.get("lastPrice")),
                    bid=_safe_float(row.get("bid")),
                    ask=_safe_float(row.get("ask")),
                    volume=_safe_int(row.get("volume")),
                    open_interest=_safe_int(row.get("openInterest")),
                    implied_volatility=iv,
                    in_the_money=bool(row.get("inTheMoney"))
                    if row.get("inTheMoney") is not None
                    else None,
                    greeks=greeks_obj,
                )
            )
        return contracts

    calls = _parse_contracts(chain.calls, "call")
    puts = _parse_contracts(chain.puts, "put")

    return OptionsChain(
        ticker=ticker.upper(),
        expiration=expiration,
        calls=calls,
        puts=puts,
    )
