"""Intelligence pipeline endpoints."""

import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request

from synesis.core.dependencies import (
    Crawl4AICrawlerDep,
    MassiveClientDep,
    SECEdgarClientDep,
    YFinanceClientDep,
)
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

router = APIRouter()

logger = get_logger(__name__)


_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,14}$")


class AnalyzeRequest(BaseModel):
    """Request body for the ticker analysis endpoint."""

    tickers: list[str] = Field(min_length=1, max_length=10)

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(t.upper().strip() for t in v))  # dedup, preserve order
        for t in cleaned:
            if not _TICKER_RE.match(t):
                raise ValueError(f"Invalid ticker: {t!r}")
        return cleaned


@router.post("/analyze")
@limiter.limit("2/minute")
async def analyze_tickers(
    request: Request,
    body: AnalyzeRequest,
    sec_edgar: SECEdgarClientDep,
    yfinance: YFinanceClientDep,
    massive: MassiveClientDep,
    crawler: Crawl4AICrawlerDep,
) -> dict[str, Any]:
    """Run the deep intelligence pipeline for specific tickers (synchronous).

    For each requested ticker, runs: ticker research + company analysis
    (fundamentals + filings) + price/technicals analysis (all parallel) →
    bull/bear debate → trader (R/R + conviction tier). Saves a markdown brief to
    `docs/kg/raw/synesis_briefs/YYYY-MM-DD-tradeideas.md`.

    Runs synchronously and returns the full compiled brief dict.
    **Expect several minutes for multi-ticker requests.**

    **Inputs (JSON body):**
    - `tickers` (list[str], 1–10): uppercase tickers, e.g. `["NVDA", "AAPL"]`.
      Validated against `^[A-Z][A-Z0-9.]{0,14}$`. Duplicates are de-duplicated
      preserving input order.

    **Returns:** Full `dict` compiled from the pipeline state:
    - `date` (str): ISO date the brief was generated for.
    - `tickers_analyzed` (list[str]): tickers that completed company analysis.
    - `trade_ideas` (list): each idea has `tickers` (list[str]), `trade_structure`
      (str, e.g. `"long NVDA"`), `thesis`, `catalyst`, `timeframe`, `key_risk`,
      `entry_price`, `target_price`, `stop_price`, `risk_reward_ratio`,
      `conviction_tier` (1=high | 2=medium | 3=speculative), `conviction_rationale`,
      `expression_note`.
    - `portfolio_note` (str): cross-ticker observations (portfolio trader mode only).
    - `debates` (list): per-ticker bull/bear debate arguments.
    - `ticker_research` (list): web + Twitter research context per ticker.
    - `company_analyses` (list): fundamentals, insider signal, analyst consensus per ticker.
    - `price_analyses` (list): technicals + options context per ticker.
    - `errors` (dict): per-node failure flags for downstream visibility.

    **Errors:**
    - `422` on invalid ticker format or empty/oversized list.
    - `500` on pipeline failure or empty result.

    **Example:**
    ```bash
    curl -X POST http://localhost:7337/api/v1/intelligence/analyze \\
      -H "Content-Type: application/json" \\
      -d '{"tickers": ["NVDA", "AMD"]}'
    ```
    Response (truncated):
    ```json
    {
      "date": "2026-05-05",
      "tickers_analyzed": ["NVDA", "AMD"],
      "trade_ideas": [
        {
          "tickers": ["NVDA"],
          "trade_structure": "long NVDA",
          "entry_price": 120.5,
          "target_price": 138.0,
          "stop_price": 113.0,
          "conviction_tier": 1
        }
      ]
    }
    ```
    """
    from synesis.config import get_settings
    from synesis.processing.intelligence.job import run_ticker_analysis

    settings = get_settings()
    twitter_key = (
        settings.twitterapi_api_key.get_secret_value() if settings.twitterapi_api_key else None
    )

    try:
        result = await run_ticker_analysis(
            tickers=body.tickers,
            sec_edgar=sec_edgar,
            yfinance=yfinance,
            massive=massive,
            crawler=crawler,
            twitter_api_key=twitter_key,
        )
    except Exception:
        logger.exception("Ticker analysis failed", tickers=body.tickers)
        raise HTTPException(status_code=500, detail="Ticker analysis failed")
    if not result.get("date"):
        raise HTTPException(status_code=500, detail="Analysis produced no results")
    return result
