"""Cache TTL and provider-tuning settings.

Split out from config.py to keep the main settings file focused on
core configuration (API keys, ingestion, notifications, trading).
"""

from pydantic import Field


class CacheTTLSettings:
    """Cache TTL settings mixed into Settings via inheritance."""

    # ── FRED ──────────────────────────────────────────────────
    fred_cache_ttl_search: int = Field(
        default=3600,
        description="Cache TTL for FRED series search results (seconds)",
    )
    fred_cache_ttl_series: int = Field(
        default=43200,
        description="Cache TTL for FRED series metadata (seconds)",
    )
    fred_cache_ttl_observations: int = Field(
        default=21600,
        description="Cache TTL for FRED observations (seconds)",
    )
    fred_cache_ttl_releases: int = Field(
        default=43200,
        description="Cache TTL for FRED releases (seconds)",
    )
    fred_cache_ttl_release_dates: int = Field(
        default=21600,
        description="Cache TTL for FRED release dates (seconds)",
    )

    # ── Massive ───────────────────────────────────────────────
    massive_cache_ttl_bars: int = Field(
        default=300,
        description="Cache TTL for Massive bars/aggregates (seconds)",
    )
    massive_cache_ttl_reference: int = Field(
        default=21600,
        description="Cache TTL for Massive ticker/reference data (seconds, 6h)",
    )
    massive_cache_ttl_static: int = Field(
        default=86400,
        description="Cache TTL for Massive static data — exchanges, types, holidays (seconds, 24h)",
    )
    massive_cache_ttl_fundamentals: int = Field(
        default=21600,
        description="Cache TTL for Massive financials/dividends/shorts (seconds, 6h)",
    )
    massive_cache_ttl_news: int = Field(
        default=900,
        description="Cache TTL for Massive news articles (seconds, 15min)",
    )
    massive_cache_ttl_indicators: int = Field(
        default=300,
        description="Cache TTL for Massive technical indicators (seconds, 5min)",
    )
    massive_cache_ttl_market_status: int = Field(
        default=60,
        description="Cache TTL for Massive market status (seconds)",
    )
    massive_cache_ttl_options: int = Field(
        default=3600,
        description="Cache TTL for Massive options contract reference (seconds, 1h)",
    )

    # ── SEC EDGAR ─────────────────────────────────────────────
    sec_edgar_cache_ttl_submissions: int = Field(
        default=3600,
        description="Cache TTL for SEC EDGAR submissions (seconds)",
    )
    sec_edgar_cache_ttl_cik_map: int = Field(
        default=86400,
        description="Cache TTL for SEC ticker→CIK mapping (seconds)",
    )
    sec_edgar_cache_ttl_company_facts: int = Field(
        default=86400,
        description="Cache TTL for SEC XBRL company facts (seconds, 24h)",
    )
    sec_edgar_cache_ttl_xbrl_frames: int = Field(
        default=86400,
        description="Cache TTL for SEC XBRL frames (seconds, 24h)",
    )
    sec_edgar_cache_ttl_filing_content: int = Field(
        default=604800,
        description="Cache TTL for SEC filing content (seconds, 7d — filings don't change)",
    )

    # ── NASDAQ ────────────────────────────────────────────────
    nasdaq_cache_ttl_earnings: int = Field(
        default=21600,
        description="Cache TTL for NASDAQ earnings calendar per date (seconds)",
    )
    nasdaq_earnings_lookahead_days: int = Field(
        default=14,
        description="Number of days to look ahead for upcoming earnings",
    )

    # ── yfinance ──────────────────────────────────────────────
    yfinance_cache_ttl_quote: int = Field(
        default=60,
        description="Cache TTL for yfinance quote snapshots (seconds)",
    )
    yfinance_cache_ttl_history: int = Field(
        default=300,
        description="Cache TTL for yfinance OHLCV history (seconds)",
    )
    yfinance_cache_ttl_options: int = Field(
        default=60,
        description="Cache TTL for yfinance options data (seconds)",
    )
    yfinance_cache_ttl_fx: int = Field(
        default=30,
        description="Cache TTL for yfinance FX rates (seconds)",
    )
    yfinance_cache_ttl_movers: int = Field(
        default=300,
        description="Cache TTL for yfinance market movers screener (seconds)",
    )
    yfinance_cache_ttl_fundamentals: int = Field(
        default=3600,
        description="Cache TTL for yfinance company fundamentals (seconds)",
    )
    yfinance_cache_ttl_analyst: int = Field(
        default=3600,
        description="Cache TTL for yfinance analyst ratings (seconds)",
    )
