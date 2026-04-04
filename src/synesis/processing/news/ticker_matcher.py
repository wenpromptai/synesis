"""Fast ticker matching from financial news text.

Matches company names and explicit $TICKER references against a static US ticker
list (data/us_tickers.json, sourced from Finnhub /stock/symbol?exchange=US).

Design priorities:
  1. Minimize false positives — better to miss a ticker than to surface a wrong one
  2. Speed — runs before any LLM call, must be <5ms per message
  3. No external dependencies — pure in-memory string matching

Matching strategy (precision-first):
  - $TICKER format: Only match when user explicitly writes $NVDA, $AAPL etc.
  - Curated names: ~80 high-profile companies hand-mapped (NVIDIA→NVDA, APPLE→AAPL).
  - Multi-word names: Algorithmically built from Finnhub descriptions, but ONLY
    multi-word phrases (≥2 words). Single-word matching is disabled because too many
    common English words are also ticker symbols (MAX, OWL, AI, EVER, etc.).

What this deliberately does NOT do:
  - Match standalone uppercase words as tickers (e.g. "IBM" in text won't match
    unless IBM is in CURATED_NAMES or preceded by $). This is intentional — the
    false positive rate from 3-4 letter uppercase words is too high in all-caps news.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from synesis.core.logging import get_logger

logger = get_logger(__name__)

_TICKERS_FILE = Path(
    os.environ.get(
        "TICKERS_FILE",
        str(Path(__file__).resolve().parents[4] / "data" / "us_tickers.json"),
    )
)

# =============================================================================
# Curated company name → ticker map
#
# This is the PRIMARY matching mechanism. Add entries when:
#   - A major company is frequently mentioned in financial news
#   - A non-public company appears often (prefix with ~)
#
# Single-word entries here (NVIDIA, APPLE, etc.) are the ONLY way to match
# company names that are a single word — algorithmic matching only does multi-word.
# =============================================================================

CURATED_NAMES: dict[str, str] = {
    # --- Mega-cap tech ---
    "APPLE": "AAPL",
    "AMAZON": "AMZN",
    "AMAZON.COM": "AMZN",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "META": "META",
    "META PLATFORMS": "META",
    "FACEBOOK": "META",
    "TESLA": "TSLA",
    "NETFLIX": "NFLX",
    "AOI": "AAOI",
    "BROADCOM": "AVGO",
    "QUALCOMM": "QCOM",
    "MICRON": "MU",
    "MARVELL": "MRVL",
    "MARVELL TECHNOLOGY": "MRVL",
    "INTEL": "INTC",
    "ORACLE": "ORCL",
    "SALESFORCE": "CRM",
    "ADOBE": "ADBE",
    "PAYPAL": "PYPL",
    "UBER": "UBER",
    "AIRBNB": "ABNB",
    "SPOTIFY": "SPOT",
    "PALANTIR": "PLTR",
    "SNOWFLAKE": "SNOW",
    "COINBASE": "COIN",
    "ROBINHOOD": "HOOD",
    "DISNEY": "DIS",
    "IBM": "IBM",
    # --- Financials ---
    "JPMORGAN": "JPM",
    "JPMORGAN CHASE": "JPM",
    "GOLDMAN": "GS",
    "GOLDMAN SACHS": "GS",
    "MORGAN STANLEY": "MS",
    "BANK OF AMERICA": "BAC",
    "CITIGROUP": "C",
    "WELLS FARGO": "WFC",
    "BLACKROCK": "BLK",
    "BERKSHIRE": "BRK.B",
    "BERKSHIRE HATHAWAY": "BRK.B",
    "VISA": "V",
    "MASTERCARD": "MA",
    # --- Industrials & defense ---
    "BOEING": "BA",
    "LOCKHEED": "LMT",
    "LOCKHEED MARTIN": "LMT",
    "RAYTHEON": "RTX",
    "NORTHROP": "NOC",
    "NORTHROP GRUMMAN": "NOC",
    "CATERPILLAR": "CAT",
    # --- Consumer ---
    "WALMART": "WMT",
    "COSTCO": "COST",
    "STARBUCKS": "SBUX",
    "MCDONALD": "MCD",
    "MCDONALDS": "MCD",
    "COCA-COLA": "KO",
    "PEPSI": "PEP",
    "PEPSICO": "PEP",
    # --- Healthcare ---
    "PFIZER": "PFE",
    "MODERNA": "MRNA",
    "ASTRAZENECA": "AZN",
    "ELI LILLY": "LLY",
    "JOHNSON & JOHNSON": "JNJ",
    "MERCK": "MRK",
    # --- Energy ---
    "EXXON": "XOM",
    "EXXONMOBIL": "XOM",
    "CHEVRON": "CVX",
    "SHELL": "SHEL",
    # --- Custom private tickers (not publicly traded, prefixed with ~) ---
    "OPENAI": "~OPENAI",
    "SPACEX": "~SPACEX",
    "STARLINK": "~SPACEX",
    "STRIPE": "~STRIPE",
    "BYTEDANCE": "~BYTEDANCE",
    "TIKTOK": "~BYTEDANCE",
    "SHEIN": "~SHEIN",
    "DATABRICKS": "~DATABRICKS",
    "ANTHROPIC": "~ANTHROPIC",
}

# =============================================================================
# Algorithmic multi-word name matching (supplements CURATED_NAMES)
#
# Built from Finnhub descriptions by stripping corporate suffixes.
# ONLY multi-word phrases are used — single words are too noisy.
# e.g. "ELI LILLY", "PALO ALTO NETWORKS", "BLUE OWL CAPITAL"
# =============================================================================

_STRIP_RE = re.compile(
    r"[-/]\s*(?:CL\s*[A-Z]|CLASS\s*[A-Z]|SER(?:IES)?\s*[A-Z0-9]+)\s*$"
    r"|[,.]?\s*(?:INC\.?|CORP\.?|CO\.?|LTD\.?|PLC|LLC|LP|NV|SA|SE|AG"
    r"|GROUP|HOLDINGS?|ENTERPRISES?|& CO\.?)\s*$",
    re.IGNORECASE,
)

# First words of company names that are too generic to be distinctive
# "GLOBAL X LITHIUM" → skip because "GLOBAL" starts too many companies
_GENERIC_FIRST_WORDS = frozenset(
    {
        "ALPHA",
        "AMERICAN",
        "ATLANTIC",
        "BITCOIN",
        "CAPITAL",
        "CLEAN",
        "CRYPTO",
        "DELTA",
        "DIGITAL",
        "EMERGING",
        "ENERGY",
        "FINANCIAL",
        "FIRST",
        "FOCUS",
        "FREEDOM",
        "FRONTIER",
        "FUND",
        "GENERAL",
        "GLOBAL",
        "GOLDEN",
        "GREEN",
        "GROWTH",
        "HEALTHCARE",
        "INTERNATIONAL",
        "INVEST",
        "INVESTMENT",
        "INVESCO",
        "ISHARES",
        "LIBERTY",
        "MARKET",
        "NATIONAL",
        "NATURAL",
        "NORTH",
        "OCEAN",
        "PACIFIC",
        "PARTNERS",
        "POWER",
        "PRECIOUS",
        "PRIME",
        "PRIVATE",
        "PROSHARES",
        "PUBLIC",
        "ROYAL",
        "SCHWAB",
        "SHARES",
        "SILVER",
        "SMART",
        "SOLAR",
        "SOUTH",
        "SPDR",
        "STANDARD",
        "STOCK",
        "STRATEGIC",
        "SUMMIT",
        "SUPER",
        "TARGET",
        "TECHNOLOGY",
        "TRADE",
        "TRUST",
        "UNITED",
        "VALUE",
        "VANGUARD",
        "WELLINGTON",
        "WISDOMTREE",
    }
)


def _clean_description(desc: str) -> str:
    """Strip corporate suffixes from a Finnhub company description.

    "NVIDIA CORP" → "NVIDIA"
    "AMAZON.COM INC" → "AMAZON.COM"
    "JPMORGAN CHASE & CO" → "JPMORGAN CHASE"
    """
    name = desc.upper().strip()
    for _ in range(3):
        cleaned = _STRIP_RE.sub("", name).strip().rstrip("&").strip()
        if cleaned == name:
            break
        name = cleaned
    return name


# =============================================================================
# Data loader (singleton)
# =============================================================================


class _TickerData:
    """Loaded ticker data. Initialized once on first match_tickers() call."""

    def __init__(self) -> None:
        self.name_to_ticker: dict[str, str] = {}
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not _TICKERS_FILE.exists():
            logger.warning("Ticker file not found", path=str(_TICKERS_FILE))
            self._loaded = True
            return

        with open(_TICKERS_FILE) as f:
            raw: dict[str, str] = json.load(f)

        self.name_to_ticker = self._build_name_map(raw)
        self._loaded = True
        logger.info("Ticker matcher loaded", name_phrases=len(self.name_to_ticker))

    @staticmethod
    def _build_name_map(raw: dict[str, str]) -> dict[str, str]:
        """Build name → ticker lookup.

        Multi-word names only (algorithmic). Single-word from CURATED_NAMES only.
        Sort by description length so canonical companies win ties:
        "APPLE INC" (AAPL) before "APPLE HOSPITALITY REIT INC" (APLE).
        """
        name_map: dict[str, str] = {}

        for sym, desc in sorted(raw.items(), key=lambda x: len(x[1])):
            name = _clean_description(desc)
            if not name or len(name) < 3:
                continue

            # Only multi-word names algorithmically
            if " " in name and name not in name_map:
                first_word = name.split()[0]
                if first_word not in _GENERIC_FIRST_WORDS:
                    name_map[name] = sym

        # Curated names override (highest priority)
        for name, ticker in CURATED_NAMES.items():
            name_map[name.upper()] = ticker

        return name_map


_data = _TickerData()

# Regex matching explicit $TICKER references (e.g. "$NVDA", "$AAPL")
_DOLLAR_TICKER_RE = re.compile(r"[$]([A-Z]{1,5})\b")


# =============================================================================
# Public API
# =============================================================================


def match_tickers(text: str) -> list[str]:
    """Match ticker symbols and company names in financial news text.

    Returns deduplicated list of matched tickers, ordered by position in text.
    Non-public companies are prefixed with ~ (e.g. ~OPENAI, ~SPACEX).

    Matching rules (precision-first):
      1. Explicit $TICKER format (e.g. "$NVDA") → always matched
      2. Curated company names (e.g. "NVIDIA" → NVDA)
      3. Multi-word names from Finnhub (e.g. "ELI LILLY" → LLY)
      Standalone uppercase words are NOT matched (too many false positives).

    Examples:
        "*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY"  → ["NVDA", "MRVL"]
        "AMAZON IN TALKS TO INVEST UP TO $50B"       → ["AMZN"]
        "$TSLA $AAPL $MSFT ALL DOWN TODAY"            → ["TSLA", "AAPL", "MSFT"]
        "JUST IN: OPENAI RAISES $122B"                → ["~OPENAI"]
        "US CPI (YOY) (MAR) ACTUAL: 2.5%"            → []
    """
    _data.ensure_loaded()
    if not _data.name_to_ticker:
        return []

    upper = text.upper()
    found: dict[str, int] = {}  # ticker → first position in text

    # 1. Explicit $TICKER format (highest confidence)
    for m in _DOLLAR_TICKER_RE.finditer(upper):
        sym = m.group(1)
        if sym not in found:
            found[sym] = m.start()

    # 2. Name matching (longest phrase first for greedy matching)
    for name, ticker in sorted(_data.name_to_ticker.items(), key=lambda x: len(x[0]), reverse=True):
        if not ticker:
            continue

        pos = upper.find(name)
        if pos < 0:
            continue

        # Word boundary check (prevent "APPLETON" matching "APPLE")
        before_ok = pos == 0 or not upper[pos - 1].isalpha()
        after_pos = pos + len(name)
        after_ok = after_pos >= len(upper) or not upper[after_pos].isalpha()

        if before_ok and after_ok and ticker not in found:
            found[ticker] = pos

    return [ticker for ticker, _ in sorted(found.items(), key=lambda x: x[1])]
