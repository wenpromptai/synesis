# Phase 2: USCompanyAnalyst + YFinance Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a USCompanyAnalyst agent that combines SEC EDGAR filings/insiders with yfinance fundamentals to produce comprehensive company analysis with deterministic scoring + LLM synthesis.

**Architecture:** Three-phase pipeline per ticker: (1) data gathering from yfinance + SEC EDGAR, (2) deterministic scoring (Piotroski F-Score, Beneish M-Score, insider cluster detection, red flags), (3) LLM synthesis via PydanticAI agent with OpenAI 5.2. New `get_fundamentals()` method on YFinanceClient provides market data. Agent lives in `processing/intelligence/specialists/`.

**Tech Stack:** PydanticAI, OpenAI 5.2 (400k context), SEC EDGAR provider, YFinance provider, Redis caching, asyncpg

---

## File Structure

| File | Responsibility |
|---|---|
| `src/synesis/providers/yfinance/models.py` | Add `CompanyFundamentals` model |
| `src/synesis/providers/yfinance/client.py` | Add `get_fundamentals()` method |
| `src/synesis/config_cache.py` | Add `yfinance_cache_ttl_fundamentals` |
| `src/synesis/processing/intelligence/__init__.py` | Package init |
| `src/synesis/processing/intelligence/models.py` | `FinancialHealthScore`, `InsiderSignal`, `RedFlag`, `CompanyAnalysis` |
| `src/synesis/processing/intelligence/specialists/__init__.py` | Package init |
| `src/synesis/processing/intelligence/specialists/scoring.py` | Piotroski F-Score, Beneish M-Score, insider cluster detection, red flag detection |
| `src/synesis/processing/intelligence/specialists/us_company.py` | PydanticAI agent: data gathering, scoring orchestration, LLM synthesis |
| `tests/unit/test_yfinance_fundamentals.py` | Tests for `get_fundamentals()` |
| `tests/unit/test_scoring.py` | Tests for deterministic scoring functions |
| `tests/unit/test_us_company_models.py` | Tests for output models |
| `tests/integration/test_us_company_agent.py` | Integration test running agent against real AXTI data |

---

### Task 1: Add `CompanyFundamentals` Model to YFinance

**Files:**
- Modify: `src/synesis/providers/yfinance/models.py`
- Create: `tests/unit/test_yfinance_fundamentals.py`

- [ ] **Step 1: Write the model test**

```python
"""Tests for CompanyFundamentals model."""

from __future__ import annotations

from synesis.providers.yfinance.models import CompanyFundamentals


def test_company_fundamentals_from_yfinance_info():
    """CompanyFundamentals parses a real yfinance .info dict."""
    info = {
        "shortName": "AXT Inc",
        "sector": "Technology",
        "industry": "Semiconductor Equipment & Materials",
        "fullTimeEmployees": 1541,
        "longBusinessSummary": "AXT, Inc. designs semiconductor substrates.",
        "marketCap": 2936508928.0,
        "beta": 1.514,
        "currentRatio": 2.723,
        "quickRatio": 1.625,
        "debtToEquity": 20.923,
        "returnOnEquity": -0.07871,
        "returnOnAssets": -0.03553,
        "grossMargins": 0.12728,
        "operatingMargins": -0.16636,
        "profitMargins": -0.2407,
        "revenueGrowth": -0.082,
        "freeCashflow": 1823375,
        "ebitda": -12868000,
        "totalCash": 120266000,
        "totalDebt": 70016000,
        "totalRevenue": 88326000,
        "sharesShort": 6168362,
        "shortPercentOfFloat": 0.1175,
        "priceToBook": 10.492,
        "priceToSalesTrailing12Months": 33.246,
        "enterpriseToEbitda": -222.087,
        "enterpriseToRevenue": 32.355,
        "forwardEps": 0.467,
        "trailingEps": -0.49,
        "targetMeanPrice": 30.75,
        "targetHighPrice": 45.0,
        "targetLowPrice": 21.0,
        "numberOfAnalystOpinions": 4,
        "heldPercentInsiders": 0.0518,
        "heldPercentInstitutions": 0.5106,
    }

    f = CompanyFundamentals.from_yfinance_info("AXTI", info)

    assert f.ticker == "AXTI"
    assert f.name == "AXT Inc"
    assert f.sector == "Technology"
    assert f.industry == "Semiconductor Equipment & Materials"
    assert f.employees == 1541
    assert f.market_cap == 2936508928.0
    assert f.beta == 1.514
    assert f.current_ratio == 2.723
    assert f.roe == -0.07871
    assert f.free_cash_flow == 1823375
    assert f.short_percent_of_float == 0.1175
    assert f.analyst_target_mean == 30.75
    assert f.held_percent_insiders == 0.0518


def test_company_fundamentals_handles_missing_fields():
    """CompanyFundamentals gracefully handles missing/None fields."""
    f = CompanyFundamentals.from_yfinance_info("FAKE", {})

    assert f.ticker == "FAKE"
    assert f.name is None
    assert f.market_cap is None
    assert f.roe is None
    assert f.free_cash_flow is None
    assert f.short_percent_of_float is None


def test_company_fundamentals_handles_nan():
    """CompanyFundamentals converts NaN to None."""
    import math
    info = {
        "marketCap": float("nan"),
        "beta": float("inf"),
        "currentRatio": 2.5,
    }
    f = CompanyFundamentals.from_yfinance_info("TEST", info)

    assert f.market_cap is None
    assert f.beta is None
    assert f.current_ratio == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_yfinance_fundamentals.py -v`
Expected: FAIL with `ImportError: cannot import name 'CompanyFundamentals'`

- [ ] **Step 3: Add CompanyFundamentals model**

Add to `src/synesis/providers/yfinance/models.py`:

```python
import math
from typing import Any


def _clean_float(val: Any) -> float | None:
    """Convert to float, return None for NaN/Inf/missing."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


class CompanyFundamentals(BaseModel):
    """Fundamental data snapshot from yfinance .info dict."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    employees: int | None = None
    business_summary: str | None = None

    # Market data
    market_cap: float | None = None
    beta: float | None = None
    total_revenue: float | None = None
    ebitda: float | None = None
    total_cash: float | None = None
    total_debt: float | None = None
    free_cash_flow: float | None = None

    # Ratios
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None

    # Valuation
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_revenue: float | None = None
    forward_eps: float | None = None
    trailing_eps: float | None = None

    # Short interest
    shares_short: int | None = None
    short_percent_of_float: float | None = None

    # Analyst
    analyst_target_mean: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    analyst_count: int | None = None

    # Ownership
    held_percent_insiders: float | None = None
    held_percent_institutions: float | None = None

    @classmethod
    def from_yfinance_info(cls, ticker: str, info: dict[str, Any]) -> CompanyFundamentals:
        """Build from a yfinance Ticker.info dict."""
        return cls(
            ticker=ticker.upper(),
            name=info.get("shortName") or info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            employees=info.get("fullTimeEmployees"),
            business_summary=info.get("longBusinessSummary"),
            market_cap=_clean_float(info.get("marketCap")),
            beta=_clean_float(info.get("beta")),
            total_revenue=_clean_float(info.get("totalRevenue")),
            ebitda=_clean_float(info.get("ebitda")),
            total_cash=_clean_float(info.get("totalCash")),
            total_debt=_clean_float(info.get("totalDebt")),
            free_cash_flow=_clean_float(info.get("freeCashflow")),
            current_ratio=_clean_float(info.get("currentRatio")),
            quick_ratio=_clean_float(info.get("quickRatio")),
            debt_to_equity=_clean_float(info.get("debtToEquity")),
            roe=_clean_float(info.get("returnOnEquity")),
            roa=_clean_float(info.get("returnOnAssets")),
            gross_margin=_clean_float(info.get("grossMargins")),
            operating_margin=_clean_float(info.get("operatingMargins")),
            profit_margin=_clean_float(info.get("profitMargins")),
            revenue_growth=_clean_float(info.get("revenueGrowth")),
            price_to_book=_clean_float(info.get("priceToBook")),
            price_to_sales=_clean_float(info.get("priceToSalesTrailing12Months")),
            ev_to_ebitda=_clean_float(info.get("enterpriseToEbitda")),
            ev_to_revenue=_clean_float(info.get("enterpriseToRevenue")),
            forward_eps=_clean_float(info.get("forwardEps")),
            trailing_eps=_clean_float(info.get("trailingEps")),
            shares_short=info.get("sharesShort"),
            short_percent_of_float=_clean_float(info.get("shortPercentOfFloat")),
            analyst_target_mean=_clean_float(info.get("targetMeanPrice")),
            analyst_target_high=_clean_float(info.get("targetHighPrice")),
            analyst_target_low=_clean_float(info.get("targetLowPrice")),
            analyst_count=info.get("numberOfAnalystOpinions"),
            held_percent_insiders=_clean_float(info.get("heldPercentInsiders")),
            held_percent_institutions=_clean_float(info.get("heldPercentInstitutions")),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_yfinance_fundamentals.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/synesis/providers/yfinance/models.py tests/unit/test_yfinance_fundamentals.py
git commit -m "feat: add CompanyFundamentals model to yfinance provider"
```

---

### Task 2: Add `get_fundamentals()` to YFinanceClient

**Files:**
- Modify: `src/synesis/providers/yfinance/client.py`
- Modify: `src/synesis/config_cache.py`
- Modify: `tests/unit/test_yfinance_fundamentals.py`

- [ ] **Step 1: Write the client method test**

Append to `tests/unit/test_yfinance_fundamentals.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from synesis.providers.yfinance.client import YFinanceClient


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    return redis


SAMPLE_INFO = {
    "shortName": "AXT Inc",
    "sector": "Technology",
    "industry": "Semiconductor Equipment & Materials",
    "marketCap": 2936508928.0,
    "beta": 1.514,
    "currentRatio": 2.723,
    "returnOnEquity": -0.07871,
    "freeCashflow": 1823375,
}


@pytest.mark.asyncio
async def test_get_fundamentals_fetches_and_caches(mock_redis):
    """get_fundamentals calls yfinance, returns CompanyFundamentals, caches result."""
    client = YFinanceClient(mock_redis)

    with patch(
        "synesis.providers.yfinance.client._fetch_quote_info",
        return_value=SAMPLE_INFO,
    ):
        result = await client.get_fundamentals("AXTI")

    assert result.ticker == "AXTI"
    assert result.market_cap == 2936508928.0
    assert result.roe == -0.07871
    mock_redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_get_fundamentals_uses_cache(mock_redis):
    """get_fundamentals returns cached result when available."""
    cached_data = CompanyFundamentals.from_yfinance_info("AXTI", SAMPLE_INFO)
    mock_redis.get = AsyncMock(
        return_value=orjson.dumps(cached_data.model_dump(mode="json"))
    )
    client = YFinanceClient(mock_redis)

    with patch(
        "synesis.providers.yfinance.client._fetch_quote_info"
    ) as mock_fetch:
        result = await client.get_fundamentals("AXTI")

    mock_fetch.assert_not_called()
    assert result.ticker == "AXTI"
    assert result.market_cap == 2936508928.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_yfinance_fundamentals.py::test_get_fundamentals_fetches_and_caches -v`
Expected: FAIL with `AttributeError: 'YFinanceClient' object has no attribute 'get_fundamentals'`

- [ ] **Step 3: Add cache TTL setting**

Add to `src/synesis/config_cache.py` after the `yfinance_cache_ttl_movers` field:

```python
    yfinance_cache_ttl_fundamentals: int = Field(
        default=3600,
        description="Cache TTL for yfinance fundamentals (seconds)",
    )
```

- [ ] **Step 4: Add get_fundamentals method and import**

Add `CompanyFundamentals` to the import block in `src/synesis/providers/yfinance/client.py`:

```python
from synesis.providers.yfinance.models import (
    CompanyFundamentals,
    EquityQuote,
    ...
)
```

Add this method to the `YFinanceClient` class (after `get_quote`):

```python
    async def get_fundamentals(self, ticker: str) -> CompanyFundamentals:
        """Get fundamental ratios and metrics for a company.

        Pulls from the same yfinance .info dict as get_quote but extracts
        a much richer set of fields: ratios, margins, valuation, short
        interest, analyst targets, ownership percentages.
        """
        settings = get_settings()
        ticker_up = ticker.upper()
        cache_key = f"{CACHE_PREFIX}:fundamentals:{ticker_up}"

        cached = await self._redis.get(cache_key)
        if cached:
            try:
                return CompanyFundamentals.model_validate(orjson.loads(cached))
            except Exception as e:
                logger.warning("Cache deserialization failed", key=cache_key, error=str(e))

        info = await asyncio.to_thread(_fetch_quote_info, ticker)
        fundamentals = CompanyFundamentals.from_yfinance_info(ticker_up, info)

        await self._redis.set(
            cache_key,
            orjson.dumps(fundamentals.model_dump(mode="json")),
            ex=settings.yfinance_cache_ttl_fundamentals,
        )
        return fundamentals
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_yfinance_fundamentals.py -v`
Expected: 5 passed

- [ ] **Step 6: Lint**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 7: Commit**

```bash
git add src/synesis/providers/yfinance/client.py src/synesis/providers/yfinance/models.py src/synesis/config_cache.py tests/unit/test_yfinance_fundamentals.py
git commit -m "feat: add get_fundamentals() to YFinanceClient"
```

---

### Task 3: Intelligence Output Models

**Files:**
- Create: `src/synesis/processing/intelligence/__init__.py`
- Create: `src/synesis/processing/intelligence/models.py`
- Create: `src/synesis/processing/intelligence/specialists/__init__.py`
- Create: `tests/unit/test_us_company_models.py`

- [ ] **Step 1: Write model validation tests**

```python
"""Tests for intelligence output models."""

from __future__ import annotations

from datetime import date

from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    FinancialHealthScore,
    InsiderSignal,
    RedFlag,
)


def test_financial_health_score_all_none():
    """FinancialHealthScore allows all-None fields for tickers with sparse data."""
    score = FinancialHealthScore(
        quarterly_eps_trend=[],
        quarterly_revenue_trend=[],
        latest_filing_period="FY2025",
    )
    assert score.piotroski_f is None
    assert score.market_cap is None


def test_financial_health_score_full():
    """FinancialHealthScore with all fields populated."""
    score = FinancialHealthScore(
        market_cap=2936508928.0,
        beta=1.514,
        current_ratio=2.723,
        quick_ratio=1.625,
        debt_to_equity=20.923,
        roe=-0.07871,
        roa=-0.03553,
        gross_margin=0.127,
        operating_margin=-0.166,
        profit_margin=-0.241,
        revenue_growth=-0.082,
        free_cash_flow=1823375,
        ebitda=-12868000,
        total_cash=120266000,
        total_debt=70016000,
        short_percent_of_float=0.1175,
        price_to_book=10.49,
        ev_to_ebitda=-222.0,
        forward_eps=0.467,
        piotroski_f=4,
        beneish_m=-2.5,
        quarterly_eps_trend=[
            {"period": "2025-12-31", "actual": -0.49, "form": "10-K"},
            {"period": "2025-09-30", "actual": -0.04, "form": "10-Q"},
        ],
        quarterly_revenue_trend=[
            {"period": "2025-12-31", "actual": 88326000, "form": "10-K"},
            {"period": "2025-09-30", "actual": 27955000, "form": "10-Q"},
        ],
        latest_filing_period="FY2025",
    )
    assert score.piotroski_f == 4
    assert score.beneish_m == -2.5


def test_insider_signal():
    """InsiderSignal from AXTI-like data (100% selling)."""
    signal = InsiderSignal(
        mspr=-1.0,
        buy_count=0,
        sell_count=10,
        total_buy_value=0.0,
        total_sell_value=15418731.84,
        cluster_detected=True,
        csuite_activity="CFO Fischer sold 89,032 shares ($4.5M) on 2026-03-12/13",
        form144_count=5,
        notable_transactions=[
            "YOUNG MORRIS S sold 125,893 shares @ $36.51 on 2026-03-09",
            "FISCHER GARY L (CFO) sold 80,776 shares @ $50.64 on 2026-03-13",
        ],
        signal="strong_sell",
    )
    assert signal.signal == "strong_sell"
    assert signal.cluster_detected is True


def test_red_flag():
    """RedFlag model."""
    flag = RedFlag(
        category="governance",
        flag="insider_selling_cluster",
        severity="critical",
        evidence="5 insiders sold within 4 days (2026-03-09 to 2026-03-13)",
    )
    assert flag.severity == "critical"


def test_company_analysis_round_trip():
    """CompanyAnalysis serializes and deserializes."""
    analysis = CompanyAnalysis(
        ticker="AXTI",
        company_name="AXT Inc",
        sector="Technology",
        industry="Semiconductor Equipment & Materials",
        analysis_date=date(2026, 4, 5),
        latest_annual_filing="10-K FY2025, filed 2026-03-17",
        financial_health=FinancialHealthScore(
            market_cap=2936508928.0,
            piotroski_f=4,
            quarterly_eps_trend=[],
            quarterly_revenue_trend=[],
            latest_filing_period="FY2025",
        ),
        insider_signal=InsiderSignal(
            mspr=-1.0,
            buy_count=0,
            sell_count=10,
            total_buy_value=0.0,
            total_sell_value=15418731.84,
            cluster_detected=True,
            csuite_activity="CFO sold heavily",
            form144_count=5,
            notable_transactions=["CFO sold 80K shares"],
            signal="strong_sell",
        ),
        red_flags=[
            RedFlag(
                category="governance",
                flag="insider_selling_cluster",
                severity="critical",
                evidence="5 insiders sold within 4 days",
            )
        ],
        business_summary="Compound semiconductor substrate maker",
        earnings_quality="Negative net income but positive FCF",
        risk_assessment="China revenue concentration, tariff exposure",
        geographic_exposure="40% China, 30% US, 30% Asia-Pacific",
        key_customers_suppliers="Top customers undisclosed, key raw material: gallium",
        insider_vs_financials="Heavy insider selling contradicts improving Q3 revenue",
        disclosure_consistency="MD&A tone cautiously optimistic despite losses",
        overall_signal="sell",
        confidence=0.72,
        primary_thesis="Insiders dumping shares despite semiconductor recovery narrative",
        key_risks=["China tariff escalation", "Gallium supply disruption", "Customer concentration"],
        monitoring_triggers=["Next earnings 2026-04-30", "Any insider buying", "China trade policy changes"],
    )
    data = analysis.model_dump(mode="json")
    restored = CompanyAnalysis.model_validate(data)
    assert restored.ticker == "AXTI"
    assert restored.confidence == 0.72
    assert len(restored.red_flags) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_us_company_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'synesis.processing.intelligence'`

- [ ] **Step 3: Create package files and models**

Create `src/synesis/processing/intelligence/__init__.py`:

```python
```

Create `src/synesis/processing/intelligence/specialists/__init__.py`:

```python
```

Create `src/synesis/processing/intelligence/models.py`:

```python
"""Output models for the intelligence pipeline.

Shared across all specialist agents and the LangGraph pipeline (Phase 3).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class FinancialHealthScore(BaseModel):
    """Mix of yfinance pre-computed ratios + XBRL multi-quarter trends."""

    # yfinance ratios
    market_cap: float | None = None
    beta: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    free_cash_flow: float | None = None
    ebitda: float | None = None
    total_cash: float | None = None
    total_debt: float | None = None
    short_percent_of_float: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    forward_eps: float | None = None

    # Computed scores (from XBRL multi-quarter data)
    piotroski_f: int | None = None
    beneish_m: float | None = None

    # XBRL quarterly trends
    quarterly_eps_trend: list[dict[str, Any]] = Field(default_factory=list)
    quarterly_revenue_trend: list[dict[str, Any]] = Field(default_factory=list)
    latest_filing_period: str = ""


class InsiderSignal(BaseModel):
    """Aggregated insider transaction analysis from SEC EDGAR."""

    mspr: float | None = None
    buy_count: int = 0
    sell_count: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    cluster_detected: bool = False
    csuite_activity: str = ""
    form144_count: int = 0
    notable_transactions: list[str] = Field(default_factory=list)
    signal: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"] = "neutral"


class RedFlag(BaseModel):
    """Individual red flag detection."""

    category: Literal["financial", "governance", "disclosure"]
    flag: str
    severity: Literal["critical", "warning", "watch"]
    evidence: str


class CompanyAnalysis(BaseModel):
    """Final USCompanyAnalyst output per ticker."""

    ticker: str
    company_name: str
    sector: str = ""
    industry: str = ""
    analysis_date: date
    latest_annual_filing: str = ""

    # Quantitative (deterministic)
    financial_health: FinancialHealthScore
    insider_signal: InsiderSignal
    red_flags: list[RedFlag] = Field(default_factory=list)

    # Qualitative (LLM from 10-K/10-Q prose)
    business_summary: str = ""
    earnings_quality: str = ""
    risk_assessment: str = ""
    geographic_exposure: str = ""
    key_customers_suppliers: str = ""

    # Cross-referenced insights
    insider_vs_financials: str = ""
    disclosure_consistency: str = ""

    # Synthesis
    overall_signal: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"] = "neutral"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_thesis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_us_company_models.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 6: Commit**

```bash
git add src/synesis/processing/intelligence/ tests/unit/test_us_company_models.py
git commit -m "feat: add intelligence pipeline output models"
```

---

### Task 4: Deterministic Scoring Module

**Files:**
- Create: `src/synesis/processing/intelligence/specialists/scoring.py`
- Create: `tests/unit/test_scoring.py`

- [ ] **Step 1: Write Piotroski F-Score tests**

```python
"""Tests for deterministic scoring functions."""

from __future__ import annotations

from synesis.processing.intelligence.specialists.scoring import (
    compute_piotroski_f,
    compute_beneish_m,
    detect_insider_cluster,
    detect_red_flags,
)


def test_piotroski_f_strong_company():
    """A profitable, improving company scores high."""
    # 9 criteria, each 0 or 1. Max = 9.
    score = compute_piotroski_f(
        net_income_current=5000000,
        operating_cf_current=6000000,
        roa_current=0.08,
        roa_previous=0.06,
        long_term_debt_current=10000000,
        long_term_debt_previous=15000000,
        current_ratio_current=2.5,
        current_ratio_previous=2.0,
        shares_outstanding_current=1000000,
        shares_outstanding_previous=1000000,
        gross_margin_current=0.45,
        gross_margin_previous=0.40,
        asset_turnover_current=0.8,
        asset_turnover_previous=0.7,
    )
    assert score == 9


def test_piotroski_f_weak_company():
    """A company failing all criteria scores 0."""
    score = compute_piotroski_f(
        net_income_current=-5000000,
        operating_cf_current=-2000000,
        roa_current=-0.08,
        roa_previous=-0.06,
        long_term_debt_current=20000000,
        long_term_debt_previous=15000000,
        current_ratio_current=0.8,
        current_ratio_previous=1.2,
        shares_outstanding_current=1200000,
        shares_outstanding_previous=1000000,
        gross_margin_current=0.20,
        gross_margin_previous=0.30,
        asset_turnover_current=0.5,
        asset_turnover_previous=0.6,
    )
    assert score == 0


def test_piotroski_f_handles_none():
    """Missing data returns None."""
    score = compute_piotroski_f(
        net_income_current=None,
        operating_cf_current=None,
        roa_current=None,
        roa_previous=None,
        long_term_debt_current=None,
        long_term_debt_previous=None,
        current_ratio_current=None,
        current_ratio_previous=None,
        shares_outstanding_current=None,
        shares_outstanding_previous=None,
        gross_margin_current=None,
        gross_margin_previous=None,
        asset_turnover_current=None,
        asset_turnover_previous=None,
    )
    assert score is None


def test_beneish_m_clean_company():
    """A company with normal metrics scores below -1.78 (not manipulating)."""
    score = compute_beneish_m(
        dsri=1.0,  # Days Sales Receivables Index
        gmi=1.0,   # Gross Margin Index
        aqi=1.0,   # Asset Quality Index
        sgi=1.1,   # Sales Growth Index
        depi=1.0,  # Depreciation Index
        sgai=1.0,  # SGA Index
        lvgi=1.0,  # Leverage Index
        tata=0.01,  # Total Accruals to Total Assets
    )
    # M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
    #     + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    assert score is not None
    assert score < -1.78  # Not a likely manipulator


def test_beneish_m_manipulator():
    """Extreme metrics push score above -1.78."""
    score = compute_beneish_m(
        dsri=2.0,
        gmi=1.5,
        aqi=1.8,
        sgi=2.0,
        depi=1.5,
        sgai=0.5,
        lvgi=1.5,
        tata=0.15,
    )
    assert score is not None
    assert score > -1.78  # Likely manipulator


def test_beneish_m_handles_none():
    """Missing components returns None."""
    score = compute_beneish_m(
        dsri=None, gmi=None, aqi=None, sgi=None,
        depi=None, sgai=None, lvgi=None, tata=None,
    )
    assert score is None


def test_detect_insider_cluster_found():
    """Detects cluster when 3+ insiders sell within 14 days."""
    transactions = [
        {"owner_name": "Alice", "transaction_date": "2026-03-10", "transaction_code": "S"},
        {"owner_name": "Bob", "transaction_date": "2026-03-11", "transaction_code": "S"},
        {"owner_name": "Charlie", "transaction_date": "2026-03-13", "transaction_code": "S"},
    ]
    assert detect_insider_cluster(transactions, window_days=14) is True


def test_detect_insider_cluster_not_found():
    """No cluster when fewer than 3 unique insiders."""
    transactions = [
        {"owner_name": "Alice", "transaction_date": "2026-03-10", "transaction_code": "S"},
        {"owner_name": "Alice", "transaction_date": "2026-03-11", "transaction_code": "S"},
        {"owner_name": "Bob", "transaction_date": "2026-03-13", "transaction_code": "S"},
    ]
    assert detect_insider_cluster(transactions, window_days=14) is False


def test_detect_insider_cluster_spread_too_wide():
    """No cluster when insiders trade more than 14 days apart."""
    transactions = [
        {"owner_name": "Alice", "transaction_date": "2026-01-10", "transaction_code": "S"},
        {"owner_name": "Bob", "transaction_date": "2026-02-15", "transaction_code": "S"},
        {"owner_name": "Charlie", "transaction_date": "2026-03-20", "transaction_code": "S"},
    ]
    assert detect_insider_cluster(transactions, window_days=14) is False


def test_detect_red_flags_late_filing():
    """Late filing alert produces a critical red flag."""
    flags = detect_red_flags(
        late_filings=[{"form_type": "NT 10-K", "filed_date": "2026-03-16"}],
        insider_transactions=[],
        financial_data={},
    )
    assert len(flags) >= 1
    late_flag = next(f for f in flags if f.flag == "late_filing")
    assert late_flag.severity == "critical"
    assert late_flag.category == "disclosure"


def test_detect_red_flags_insider_selling_cluster():
    """Insider selling cluster produces a governance red flag."""
    flags = detect_red_flags(
        late_filings=[],
        insider_transactions=[
            {"owner_name": "A", "transaction_date": "2026-03-10", "transaction_code": "S"},
            {"owner_name": "B", "transaction_date": "2026-03-11", "transaction_code": "S"},
            {"owner_name": "C", "transaction_date": "2026-03-12", "transaction_code": "S"},
        ],
        financial_data={},
    )
    cluster_flag = next((f for f in flags if f.flag == "insider_selling_cluster"), None)
    assert cluster_flag is not None
    assert cluster_flag.severity == "critical"
    assert cluster_flag.category == "governance"


def test_detect_red_flags_cash_flow_divergence():
    """Positive net income but negative operating CF is a financial red flag."""
    flags = detect_red_flags(
        late_filings=[],
        insider_transactions=[],
        financial_data={
            "net_income": 5000000,
            "operating_cf": -2000000,
        },
    )
    cf_flag = next((f for f in flags if f.flag == "cash_flow_divergence"), None)
    assert cf_flag is not None
    assert cf_flag.severity == "warning"
    assert cf_flag.category == "financial"


def test_detect_red_flags_none_when_clean():
    """No flags for a clean company."""
    flags = detect_red_flags(
        late_filings=[],
        insider_transactions=[],
        financial_data={
            "net_income": 5000000,
            "operating_cf": 8000000,
        },
    )
    assert len(flags) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement scoring module**

Create `src/synesis/processing/intelligence/specialists/scoring.py`:

```python
"""Deterministic scoring for USCompanyAnalyst.

All functions are pure — no I/O, no LLM. They take pre-fetched data
and return numeric scores or structured red flags.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from synesis.processing.intelligence.models import RedFlag

# ── Piotroski F-Score ────────────────────────────────────────────


def compute_piotroski_f(
    net_income_current: float | None,
    operating_cf_current: float | None,
    roa_current: float | None,
    roa_previous: float | None,
    long_term_debt_current: float | None,
    long_term_debt_previous: float | None,
    current_ratio_current: float | None,
    current_ratio_previous: float | None,
    shares_outstanding_current: float | None,
    shares_outstanding_previous: float | None,
    gross_margin_current: float | None,
    gross_margin_previous: float | None,
    asset_turnover_current: float | None,
    asset_turnover_previous: float | None,
) -> int | None:
    """Compute Piotroski F-Score (0-9) from financial data.

    Returns None if insufficient data (all inputs are None).
    """
    criteria: list[bool | None] = [
        # Profitability (4 criteria)
        net_income_current > 0 if net_income_current is not None else None,
        operating_cf_current > 0 if operating_cf_current is not None else None,
        (roa_current > roa_previous)
        if roa_current is not None and roa_previous is not None
        else None,
        (operating_cf_current > net_income_current)
        if operating_cf_current is not None and net_income_current is not None
        else None,
        # Leverage / Liquidity (3 criteria)
        (long_term_debt_current < long_term_debt_previous)
        if long_term_debt_current is not None and long_term_debt_previous is not None
        else None,
        (current_ratio_current > current_ratio_previous)
        if current_ratio_current is not None and current_ratio_previous is not None
        else None,
        (shares_outstanding_current <= shares_outstanding_previous)
        if shares_outstanding_current is not None and shares_outstanding_previous is not None
        else None,
        # Operating efficiency (2 criteria)
        (gross_margin_current > gross_margin_previous)
        if gross_margin_current is not None and gross_margin_previous is not None
        else None,
        (asset_turnover_current > asset_turnover_previous)
        if asset_turnover_current is not None and asset_turnover_previous is not None
        else None,
    ]
    valid = [c for c in criteria if c is not None]
    if not valid:
        return None
    return sum(valid)


# ── Beneish M-Score ──────────────────────────────────────────────


def compute_beneish_m(
    dsri: float | None,
    gmi: float | None,
    aqi: float | None,
    sgi: float | None,
    depi: float | None,
    sgai: float | None,
    lvgi: float | None,
    tata: float | None,
) -> float | None:
    """Compute Beneish M-Score for earnings manipulation detection.

    M > -1.78 suggests likely manipulation.
    Requires 2 years of financial data to compute input indices.
    Returns None if any component is None.
    """
    components = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    if any(c is None for c in components):
        return None
    assert all(c is not None for c in components)  # for type narrowing

    return (
        -4.84
        + 0.920 * dsri  # type: ignore[operator]
        + 0.528 * gmi  # type: ignore[operator]
        + 0.404 * aqi  # type: ignore[operator]
        + 0.892 * sgi  # type: ignore[operator]
        + 0.115 * depi  # type: ignore[operator]
        - 0.172 * sgai  # type: ignore[operator]
        + 4.679 * tata  # type: ignore[operator]
        - 0.327 * lvgi  # type: ignore[operator]
    )


# ── Insider Cluster Detection ────────────────────────────────────


def detect_insider_cluster(
    transactions: list[dict[str, Any]],
    window_days: int = 14,
    min_insiders: int = 3,
) -> bool:
    """Detect if 3+ unique insiders traded in the same direction within a window.

    Args:
        transactions: List of dicts with owner_name, transaction_date, transaction_code.
        window_days: Rolling window size in days.
        min_insiders: Minimum unique insiders to qualify as a cluster.
    """
    if len(transactions) < min_insiders:
        return False

    dated = []
    for t in transactions:
        try:
            d = date.fromisoformat(str(t["transaction_date"]))
            dated.append((d, t["owner_name"]))
        except (KeyError, ValueError):
            continue

    dated.sort(key=lambda x: x[0])
    window = timedelta(days=window_days)

    for i, (d_start, _) in enumerate(dated):
        names_in_window = set()
        for d_other, name in dated[i:]:
            if d_other - d_start <= window:
                names_in_window.add(name)
        if len(names_in_window) >= min_insiders:
            return True

    return False


# ── Red Flag Detection ───────────────────────────────────────────


def detect_red_flags(
    late_filings: list[dict[str, Any]],
    insider_transactions: list[dict[str, Any]],
    financial_data: dict[str, Any],
) -> list[RedFlag]:
    """Detect red flags from multiple data sources.

    Args:
        late_filings: List of LateFilingAlert-like dicts.
        insider_transactions: List of open-market insider transactions.
        financial_data: Dict with keys like net_income, operating_cf.
    """
    flags: list[RedFlag] = []

    # Late filing alerts
    for lf in late_filings:
        flags.append(
            RedFlag(
                category="disclosure",
                flag="late_filing",
                severity="critical",
                evidence=f"{lf.get('form_type', 'NT')} filed on {lf.get('filed_date', 'unknown')}",
            )
        )

    # Insider selling cluster
    sells = [t for t in insider_transactions if t.get("transaction_code") == "S"]
    if detect_insider_cluster(sells):
        names = sorted({t["owner_name"] for t in sells})
        flags.append(
            RedFlag(
                category="governance",
                flag="insider_selling_cluster",
                severity="critical",
                evidence=f"{len(names)} insiders selling: {', '.join(names[:5])}",
            )
        )

    # Cash flow divergence: positive net income but negative operating CF
    net_income = financial_data.get("net_income")
    operating_cf = financial_data.get("operating_cf")
    if (
        net_income is not None
        and operating_cf is not None
        and net_income > 0
        and operating_cf < 0
    ):
        flags.append(
            RedFlag(
                category="financial",
                flag="cash_flow_divergence",
                severity="warning",
                evidence=f"Net income ${net_income:,.0f} but operating CF ${operating_cf:,.0f}",
            )
        )

    return flags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: 12 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 6: Commit**

```bash
git add src/synesis/processing/intelligence/specialists/scoring.py tests/unit/test_scoring.py
git commit -m "feat: add deterministic scoring (Piotroski, Beneish, insider clusters, red flags)"
```

---

### Task 5: USCompanyAnalyst Agent

**Files:**
- Create: `src/synesis/processing/intelligence/specialists/us_company.py`

This is the main agent module. It orchestrates data gathering, scoring, and LLM synthesis.

- [ ] **Step 1: Create the agent module**

Create `src/synesis/processing/intelligence/specialists/us_company.py`:

```python
"""USCompanyAnalyst — comprehensive US company analysis via SEC EDGAR + yfinance.

Three-phase pipeline per ticker:
1. Data gathering (no LLM): yfinance fundamentals + EDGAR XBRL/insiders/filings
2. Deterministic scoring (no LLM): Piotroski, Beneish, insider clusters, red flags
3. LLM synthesis: PydanticAI agent interprets scores + filing prose → CompanyAnalysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext

from synesis.core.logging import get_logger
from synesis.processing.common.llm import create_model
from synesis.processing.intelligence.models import (
    CompanyAnalysis,
    FinancialHealthScore,
    InsiderSignal,
    RedFlag,
)
from synesis.processing.intelligence.specialists.scoring import (
    compute_beneish_m,
    compute_piotroski_f,
    detect_insider_cluster,
    detect_red_flags,
)

if TYPE_CHECKING:
    from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider
    from synesis.providers.sec_edgar.client import SECEdgarClient
    from synesis.providers.yfinance.client import YFinanceClient

logger = get_logger(__name__)


@dataclass
class USCompanyDeps:
    """Dependencies for USCompanyAnalyst."""

    sec_edgar: SECEdgarClient
    yfinance: YFinanceClient
    crawler: Crawl4AICrawlerProvider | None = None
    current_date: date = field(default_factory=lambda: datetime.now(UTC).date())


# ── Phase 1: Data Gathering ──────────────────────────────────────


async def _gather_yfinance(yf: YFinanceClient, ticker: str) -> dict[str, Any]:
    """Fetch yfinance fundamentals + quote."""
    fundamentals = await yf.get_fundamentals(ticker)
    quote = await yf.get_quote(ticker)
    return {
        "fundamentals": fundamentals,
        "quote": quote,
    }


async def _gather_edgar_xbrl(
    edgar: SECEdgarClient, ticker: str
) -> dict[str, Any]:
    """Fetch XBRL multi-quarter data."""
    eps = await edgar.get_historical_eps(ticker, limit=8)
    revenue = await edgar.get_historical_revenue(ticker, limit=8)

    facts = await edgar.get_company_facts(
        ticker,
        concepts=[
            "Assets",
            "Liabilities",
            "StockholdersEquity",
            "NetIncomeLoss",
            "OperatingIncomeLoss",
            "GrossProfit",
            "CostOfRevenue",
            "NetCashProvidedByUsedInOperatingActivities",
            "CommonStockSharesOutstanding",
            "LongTermDebt",
        ],
        limit=4,
    )

    return {
        "eps_history": eps,
        "revenue_history": revenue,
        "facts": facts,
    }


async def _gather_edgar_insiders(
    edgar: SECEdgarClient, ticker: str
) -> dict[str, Any]:
    """Fetch insider transactions, sentiment, Form 144."""
    transactions = await edgar.get_insider_transactions(ticker, limit=20, codes=["P", "S"])
    sentiment = await edgar.get_insider_sentiment(ticker)
    form144 = await edgar.get_form144_filings(ticker, limit=10)
    late_filings = await edgar.get_late_filing_alerts(ticker)

    return {
        "transactions": transactions,
        "sentiment": sentiment,
        "form144": form144,
        "late_filings": late_filings,
    }


async def _gather_edgar_filings(
    edgar: SECEdgarClient,
    ticker: str,
    crawler: Crawl4AICrawlerProvider | None,
) -> dict[str, Any]:
    """Fetch latest 10-K and 10-Q filing content."""
    filings_10k = await edgar.get_filings(ticker, form_types=["10-K"], limit=1)
    filings_10q = await edgar.get_filings(ticker, form_types=["10-Q"], limit=1)

    content_10k = None
    content_10q = None
    filing_10k_meta = None
    filing_10q_meta = None

    if filings_10k:
        filing_10k_meta = filings_10k[0]
        content_10k = await edgar.get_filing_content(filings_10k[0].url, crawler=crawler)

    if filings_10q:
        filing_10q_meta = filings_10q[0]
        content_10q = await edgar.get_filing_content(filings_10q[0].url, crawler=crawler)

    return {
        "content_10k": content_10k,
        "content_10q": content_10q,
        "filing_10k": filing_10k_meta,
        "filing_10q": filing_10q_meta,
    }


# ── Phase 2: Deterministic Scoring ──────────────────────────────


def _build_financial_health(
    yf_data: dict[str, Any],
    xbrl_data: dict[str, Any],
) -> FinancialHealthScore:
    """Build FinancialHealthScore from yfinance + XBRL data."""
    fundamentals = yf_data["fundamentals"]
    facts = xbrl_data.get("facts")

    # Extract XBRL values for scoring (current and previous period)
    def _xbrl_val(concept: str, index: int = 0) -> float | None:
        if not facts:
            return None
        matches = [f for f in facts.facts if f.concept == concept]
        if len(matches) > index:
            return matches[index].value
        return None

    # Piotroski inputs from XBRL (current = index 0, previous = index 1)
    piotroski = compute_piotroski_f(
        net_income_current=_xbrl_val("NetIncomeLoss", 0),
        operating_cf_current=_xbrl_val("NetCashProvidedByUsedInOperatingActivities", 0),
        roa_current=fundamentals.roa,
        roa_previous=None,  # Would need prior-year yfinance data
        long_term_debt_current=_xbrl_val("LongTermDebt", 0),
        long_term_debt_previous=_xbrl_val("LongTermDebt", 1),
        current_ratio_current=fundamentals.current_ratio,
        current_ratio_previous=None,
        shares_outstanding_current=_xbrl_val("CommonStockSharesOutstanding", 0),
        shares_outstanding_previous=_xbrl_val("CommonStockSharesOutstanding", 1),
        gross_margin_current=fundamentals.gross_margin,
        gross_margin_previous=None,
        asset_turnover_current=None,  # Revenue/Assets, could compute
        asset_turnover_previous=None,
    )

    # Beneish M-Score needs 2 years of comparative data — skip if unavailable
    # Full implementation would compute DSRI, GMI, AQI, etc. from XBRL trends
    beneish = None

    eps_history = xbrl_data.get("eps_history", [])
    revenue_history = xbrl_data.get("revenue_history", [])

    latest_period = ""
    if eps_history:
        frame = eps_history[0].get("frame", "")
        latest_period = frame if frame else eps_history[0].get("period", "")

    return FinancialHealthScore(
        market_cap=fundamentals.market_cap,
        beta=fundamentals.beta,
        current_ratio=fundamentals.current_ratio,
        quick_ratio=fundamentals.quick_ratio,
        debt_to_equity=fundamentals.debt_to_equity,
        roe=fundamentals.roe,
        roa=fundamentals.roa,
        gross_margin=fundamentals.gross_margin,
        operating_margin=fundamentals.operating_margin,
        profit_margin=fundamentals.profit_margin,
        revenue_growth=fundamentals.revenue_growth,
        free_cash_flow=fundamentals.free_cash_flow,
        ebitda=fundamentals.ebitda,
        total_cash=fundamentals.total_cash,
        total_debt=fundamentals.total_debt,
        short_percent_of_float=fundamentals.short_percent_of_float,
        price_to_book=fundamentals.price_to_book,
        ev_to_ebitda=fundamentals.ev_to_ebitda,
        forward_eps=fundamentals.forward_eps,
        piotroski_f=piotroski,
        beneish_m=beneish,
        quarterly_eps_trend=eps_history,
        quarterly_revenue_trend=revenue_history,
        latest_filing_period=latest_period,
    )


def _build_insider_signal(insider_data: dict[str, Any]) -> InsiderSignal:
    """Build InsiderSignal from EDGAR insider data."""
    transactions = insider_data["transactions"]
    sentiment = insider_data.get("sentiment") or {}
    form144 = insider_data.get("form144", [])

    # Convert InsiderTransaction models to dicts for cluster detection
    txn_dicts = [
        {
            "owner_name": t.owner_name,
            "transaction_date": str(t.transaction_date),
            "transaction_code": t.transaction_code,
        }
        for t in transactions
    ]

    sells = [t for t in txn_dicts if t["transaction_code"] == "S"]
    buys = [t for t in txn_dicts if t["transaction_code"] == "P"]
    cluster = detect_insider_cluster(sells) or detect_insider_cluster(buys)

    # C-suite activity summary
    csuite_titles = {"CEO", "CFO", "COO", "CTO", "President"}
    csuite_txns = [
        t for t in transactions
        if any(title in (t.owner_relationship or "") for title in csuite_titles)
    ]
    csuite_summary = ""
    if csuite_txns:
        parts = []
        for t in csuite_txns[:3]:
            action = "bought" if t.transaction_code == "P" else "sold"
            price_str = f" @ ${t.price_per_share:.2f}" if t.price_per_share else ""
            parts.append(
                f"{t.owner_name} ({t.owner_relationship}) {action} "
                f"{int(t.shares):,} shares{price_str} on {t.transaction_date}"
            )
        csuite_summary = "; ".join(parts)

    # Notable transactions (top by dollar value)
    notable = []
    for t in sorted(
        transactions,
        key=lambda x: (x.shares or 0) * (x.price_per_share or 0),
        reverse=True,
    )[:5]:
        action = "bought" if t.transaction_code == "P" else "sold"
        price_str = f" @ ${t.price_per_share:.2f}" if t.price_per_share else ""
        notable.append(
            f"{t.owner_name} ({t.owner_relationship}) {action} "
            f"{int(t.shares):,} shares{price_str} on {t.transaction_date}"
        )

    # Determine signal from MSPR
    mspr = sentiment.get("mspr")
    if mspr is not None:
        if mspr >= 0.5:
            signal = "strong_buy"
        elif mspr >= 0.1:
            signal = "buy"
        elif mspr <= -0.5:
            signal = "strong_sell"
        elif mspr <= -0.1:
            signal = "sell"
        else:
            signal = "neutral"
    else:
        signal = "neutral"

    return InsiderSignal(
        mspr=mspr,
        buy_count=sentiment.get("buy_count", 0),
        sell_count=sentiment.get("sell_count", 0),
        total_buy_value=sentiment.get("total_buy_value", 0.0),
        total_sell_value=sentiment.get("total_sell_value", 0.0),
        cluster_detected=cluster,
        csuite_activity=csuite_summary,
        form144_count=len(form144),
        notable_transactions=notable,
        signal=signal,
    )


# ── Phase 3: LLM Synthesis ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior equity research analyst specializing in fundamental analysis of US public companies.
You combine quantitative rigor with qualitative judgment from SEC filings.

Today's date: {current_date}

## Your Analytical Framework

1. EARNINGS QUALITY (most important)
   - Cash flow from operations vs reported net income (accrual quality)
   - Revenue recognition patterns: is growth real or pulled forward?
   - Non-GAAP vs GAAP divergence: are adjustments growing?

2. FINANCIAL STRENGTH
   - Pre-computed scores are provided in the quantitative data. Interpret them in context.
   - Piotroski F-Score: 7-9 strong, 4-6 moderate, 0-3 weak
   - Beneish M-Score: >-1.78 suggests possible earnings manipulation (if available)

3. INSIDER CROSS-REFERENCING (your unique edge)
   - Compare insider buying/selling patterns against financial trends
   - C-suite buying during price weakness = strong conviction signal
   - C-suite selling during growth claims = potential red flag
   - Cluster buying/selling (3+ insiders within 14 days) = high-signal event

4. RED FLAG DETECTION
   - Late filings (NT 10-K / NT 10-Q)
   - Cash flow divergence (profit but no cash)
   - Insider selling clusters before negative announcements
   - Expanding risk factors without explanation

5. FILING PROSE ANALYSIS
   - Extract geographic exposure, customer/supplier concentration
   - Assess competitive moat and business quality from 10-K descriptions
   - Compare MD&A tone against actual financial numbers
   - Newer filings should be weighted MORE heavily than older ones

## Rules
- NEVER fabricate financial data. If unavailable, say "not available"
- Always cite which filing period (Q1 2025, FY 2024) data comes from
- Cross-reference at least 2 data points before making claims
- If insider activity contradicts financial trends, flag this prominently
- Confidence calibration:
  90-100%: Overwhelming evidence, multiple confirming signals
  70-89%: Strong evidence with minor uncertainties
  50-69%: Mixed signals, reasonable arguments both ways
  <50%: Insufficient data or highly conflicting signals
"""


def _build_user_prompt(
    ticker: str,
    yf_data: dict[str, Any],
    financial_health: FinancialHealthScore,
    insider_signal: InsiderSignal,
    red_flags: list[RedFlag],
    filing_data: dict[str, Any],
) -> str:
    """Build the user prompt with all gathered data."""
    fundamentals = yf_data["fundamentals"]
    quote = yf_data["quote"]

    sections = []

    # Company overview
    sections.append(f"# Analysis Request: {ticker}")
    if fundamentals.name:
        sections.append(f"**Company:** {fundamentals.name}")
    if fundamentals.sector:
        sections.append(f"**Sector:** {fundamentals.sector} / {fundamentals.industry}")
    if quote.last:
        sections.append(f"**Current Price:** ${quote.last:.2f}")

    # Quantitative scores
    sections.append("\n## Pre-Computed Financial Health")
    sections.append(financial_health.model_dump_json(indent=2))

    # Insider data
    sections.append("\n## Insider Activity")
    sections.append(insider_signal.model_dump_json(indent=2))

    # Red flags
    if red_flags:
        sections.append("\n## Detected Red Flags")
        for rf in red_flags:
            sections.append(f"- **[{rf.severity.upper()}] {rf.flag}** ({rf.category}): {rf.evidence}")
    else:
        sections.append("\n## Detected Red Flags\nNone detected.")

    # Filing prose
    if filing_data.get("content_10k"):
        meta = filing_data.get("filing_10k")
        label = f"10-K (filed {meta.filed_date}, period ending {meta.report_date})" if meta else "10-K"
        sections.append(f"\n## Latest Annual Filing: {label}")
        # Truncate if extremely long, but with 400k context this is rarely needed
        content = filing_data["content_10k"]
        if len(content) > 300000:
            content = content[:300000] + "\n\n[... truncated for length ...]"
        sections.append(content)

    if filing_data.get("content_10q"):
        meta = filing_data.get("filing_10q")
        label = f"10-Q (filed {meta.filed_date}, period ending {meta.report_date})" if meta else "10-Q"
        sections.append(f"\n## Latest Quarterly Filing: {label}")
        content = filing_data["content_10q"]
        if len(content) > 200000:
            content = content[:200000] + "\n\n[... truncated for length ...]"
        sections.append(content)

    sections.append(
        "\n## Instructions\n"
        "Analyze this company and produce a CompanyAnalysis. "
        "Fill in ALL qualitative fields (business_summary, earnings_quality, "
        "risk_assessment, geographic_exposure, key_customers_suppliers, "
        "insider_vs_financials, disclosure_consistency) plus the synthesis "
        "fields (overall_signal, confidence, primary_thesis, key_risks, "
        "monitoring_triggers). Use the filing prose for qualitative insights "
        "and cross-reference against the quantitative data."
    )

    return "\n".join(sections)


# ── Agent Definition ─────────────────────────────────────────────

# LLM output — just the qualitative + synthesis fields
# The quantitative fields are pre-computed and merged in after
from pydantic import BaseModel, Field
from typing import Literal


class _LLMAnalysisOutput(BaseModel):
    """Fields the LLM fills in. Merged with pre-computed data to form CompanyAnalysis."""

    business_summary: str
    earnings_quality: str
    risk_assessment: str
    geographic_exposure: str
    key_customers_suppliers: str
    insider_vs_financials: str
    disclosure_consistency: str
    overall_signal: Literal["strong_buy", "buy", "neutral", "sell", "strong_sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    primary_thesis: str
    key_risks: list[str]
    monitoring_triggers: list[str]


_agent = Agent(
    model=create_model(tier="vsmart"),
    output_type=_LLMAnalysisOutput,
    system_prompt=SYSTEM_PROMPT,
)


# ── Public API ───────────────────────────────────────────────────


async def analyze_company(
    ticker: str,
    deps: USCompanyDeps,
) -> CompanyAnalysis:
    """Run the full three-phase analysis pipeline for a single ticker.

    Args:
        ticker: US equity ticker symbol.
        deps: Injected provider clients.

    Returns:
        CompanyAnalysis with quantitative scores + LLM qualitative synthesis.
    """
    import asyncio

    ticker = ticker.upper()
    logger.info("Starting USCompanyAnalyst", ticker=ticker)

    # Phase 1: Gather data concurrently
    yf_task = asyncio.create_task(_gather_yfinance(deps.yfinance, ticker))
    xbrl_task = asyncio.create_task(_gather_edgar_xbrl(deps.sec_edgar, ticker))
    insider_task = asyncio.create_task(_gather_edgar_insiders(deps.sec_edgar, ticker))
    filing_task = asyncio.create_task(
        _gather_edgar_filings(deps.sec_edgar, ticker, deps.crawler)
    )

    yf_data, xbrl_data, insider_data, filing_data = await asyncio.gather(
        yf_task, xbrl_task, insider_task, filing_task
    )

    logger.info("Phase 1 complete: data gathered", ticker=ticker)

    # Phase 2: Deterministic scoring
    financial_health = _build_financial_health(yf_data, xbrl_data)
    insider_signal = _build_insider_signal(insider_data)

    # Red flags from late filings + insider data + financial data
    late_filing_dicts = [
        {"form_type": lf.form_type, "filed_date": str(lf.filed_date)}
        for lf in insider_data.get("late_filings", [])
    ]
    txn_dicts = [
        {
            "owner_name": t.owner_name,
            "transaction_date": str(t.transaction_date),
            "transaction_code": t.transaction_code,
        }
        for t in insider_data["transactions"]
    ]

    fundamentals = yf_data["fundamentals"]
    financial_dict = {}
    if xbrl_data.get("facts"):
        ni_facts = [f for f in xbrl_data["facts"].facts if f.concept == "NetIncomeLoss"]
        cf_facts = [
            f for f in xbrl_data["facts"].facts
            if f.concept == "NetCashProvidedByUsedInOperatingActivities"
        ]
        if ni_facts:
            financial_dict["net_income"] = ni_facts[0].value
        if cf_facts:
            financial_dict["operating_cf"] = cf_facts[0].value

    red_flags = detect_red_flags(late_filing_dicts, txn_dicts, financial_dict)

    logger.info(
        "Phase 2 complete: scoring done",
        ticker=ticker,
        piotroski=financial_health.piotroski_f,
        insider_signal=insider_signal.signal,
        red_flags=len(red_flags),
    )

    # Phase 3: LLM synthesis
    user_prompt = _build_user_prompt(
        ticker, yf_data, financial_health, insider_signal, red_flags, filing_data
    )

    result = await _agent.run(
        user_prompt.format(current_date=deps.current_date),
    )
    llm_output = result.output

    # Get company info for metadata
    company_info = await deps.sec_edgar.get_company_info(ticker)
    company_name = (
        company_info.name if company_info else (fundamentals.name or ticker)
    )

    filing_10k = filing_data.get("filing_10k")
    latest_filing = ""
    if filing_10k:
        latest_filing = f"10-K FY{filing_10k.report_date.year if filing_10k.report_date else '?'}, filed {filing_10k.filed_date}"

    logger.info("Phase 3 complete: LLM synthesis done", ticker=ticker)

    return CompanyAnalysis(
        ticker=ticker,
        company_name=company_name,
        sector=fundamentals.sector or "",
        industry=fundamentals.industry or "",
        analysis_date=deps.current_date,
        latest_annual_filing=latest_filing,
        financial_health=financial_health,
        insider_signal=insider_signal,
        red_flags=red_flags,
        business_summary=llm_output.business_summary,
        earnings_quality=llm_output.earnings_quality,
        risk_assessment=llm_output.risk_assessment,
        geographic_exposure=llm_output.geographic_exposure,
        key_customers_suppliers=llm_output.key_customers_suppliers,
        insider_vs_financials=llm_output.insider_vs_financials,
        disclosure_consistency=llm_output.disclosure_consistency,
        overall_signal=llm_output.overall_signal,
        confidence=llm_output.confidence,
        primary_thesis=llm_output.primary_thesis,
        key_risks=llm_output.key_risks,
        monitoring_triggers=llm_output.monitoring_triggers,
    )
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 3: Commit**

```bash
git add src/synesis/processing/intelligence/specialists/us_company.py
git commit -m "feat: add USCompanyAnalyst agent with 3-phase pipeline"
```

---

### Task 6: Integration Test with AXTI

**Files:**
- Create: `tests/integration/test_us_company_agent.py`

- [ ] **Step 1: Create integration test**

```python
"""Integration test for USCompanyAnalyst against real AXTI data.

Requires: Redis, SEC EDGAR (no key), yfinance (no key), OpenAI API key.
Run with: uv run pytest tests/integration/test_us_company_agent.py -v -m integration
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest
from redis.asyncio import Redis

from synesis.config import get_settings
from synesis.processing.intelligence.models import CompanyAnalysis
from synesis.processing.intelligence.specialists.us_company import (
    USCompanyDeps,
    analyze_company,
)
from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.yfinance.client import YFinanceClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_axti():
    """Full pipeline run against AXTI — verifies all phases complete."""
    settings = get_settings()
    redis = Redis.from_url(str(settings.redis_url))

    try:
        edgar = SECEdgarClient(redis)
        yfinance = YFinanceClient(redis)

        deps = USCompanyDeps(
            sec_edgar=edgar,
            yfinance=yfinance,
            crawler=None,  # Uses fallback HTML-to-text
            current_date=date.today(),
        )

        result = await analyze_company("AXTI", deps)

        # Verify structure
        assert isinstance(result, CompanyAnalysis)
        assert result.ticker == "AXTI"
        assert result.company_name  # Non-empty
        assert result.sector  # Non-empty

        # Verify quantitative data populated
        assert result.financial_health.market_cap is not None
        assert result.financial_health.market_cap > 0
        assert len(result.financial_health.quarterly_eps_trend) > 0

        # Verify insider data (AXTI has active insider selling)
        assert result.insider_signal.sell_count > 0

        # Verify LLM synthesis produced content
        assert len(result.business_summary) > 50
        assert len(result.primary_thesis) > 20
        assert result.confidence > 0
        assert result.overall_signal in ("strong_buy", "buy", "neutral", "sell", "strong_sell")
        assert len(result.key_risks) > 0

        print(f"\n{'='*60}")
        print(f"USCompanyAnalyst Result: {result.ticker}")
        print(f"{'='*60}")
        print(f"Signal: {result.overall_signal} (confidence: {result.confidence:.0%})")
        print(f"Thesis: {result.primary_thesis}")
        print(f"Piotroski F: {result.financial_health.piotroski_f}")
        print(f"Insider: {result.insider_signal.signal} (MSPR: {result.insider_signal.mspr})")
        print(f"Red flags: {len(result.red_flags)}")
        for rf in result.red_flags:
            print(f"  [{rf.severity}] {rf.flag}: {rf.evidence}")
        print(f"Key risks: {result.key_risks}")
        print(f"Monitoring: {result.monitoring_triggers}")

    finally:
        await edgar.close()
        await redis.aclose()
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/integration/test_us_company_agent.py -v -m integration`
Expected: PASS (may take 30-90 seconds due to filing content fetch + LLM call)

- [ ] **Step 3: Fix any issues discovered during integration test**

Common issues to watch for:
- `_agent` initialization at module level may fail if settings aren't loaded yet — may need to move to lazy init
- Filing content may be very large — check if OpenAI 5.2 handles it
- XBRL fact ordering may differ from expected — `_xbrl_val` should handle gracefully

- [ ] **Step 4: Lint and commit**

Run: `uv run ruff check --fix . && uv run ruff format .`

```bash
git add tests/integration/test_us_company_agent.py
git commit -m "test: add integration test for USCompanyAnalyst with AXTI"
```

---

### Task 7: Final Lint, Type Check, and Full Test Run

- [ ] **Step 1: Run full lint**

Run: `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 2: Run type check**

Run: `uv run mypy src/synesis/processing/intelligence/ src/synesis/providers/yfinance/`

- [ ] **Step 3: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`
Expected: All pass, no regressions

- [ ] **Step 4: Fix any issues**

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: lint and type fixes for Phase 2"
```
