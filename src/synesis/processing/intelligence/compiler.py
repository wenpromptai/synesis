"""Brief compiler — deterministic assembly of pipeline outputs.

No LLM. Collects all agent outputs from state and assembles them into
a structured brief with debate arguments grouped by ticker.
"""

from __future__ import annotations

from typing import Any


def compile_brief(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble an intelligence brief from pipeline state.

    Args:
        state: The full IntelligenceState dict.

    Returns:
        Structured brief dict with macro regime, debates per ticker,
        company analyses, price analyses, and macro themes.
    """
    social = state.get("social_analysis", {})
    news = state.get("news_analysis", {})
    companies = state.get("company_analyses", [])
    valid_companies = [c for c in companies if not c.get("error")]
    prices = state.get("price_analyses", [])
    valid_prices = [p for p in prices if not p.get("error")]
    macro = state.get("macro_view", {})

    # Extract bull + bear arguments by ticker (last round wins — overwrites earlier rounds)
    bull_analyses = state.get("bull_analyses", [])
    bear_analyses = state.get("bear_analyses", [])

    bull_by_ticker: dict[str, dict[str, Any]] = {}
    for item in sorted(bull_analyses, key=lambda x: x.get("round", 0)):
        if not item.get("error") and item.get("ticker"):
            bull_by_ticker[item["ticker"]] = item

    bear_by_ticker: dict[str, dict[str, Any]] = {}
    for item in sorted(bear_analyses, key=lambda x: x.get("round", 0)):
        if not item.get("error") and item.get("ticker"):
            bear_by_ticker[item["ticker"]] = item

    # Build debates list — one entry per ticker that has at least one side
    all_tickers = sorted(set(bull_by_ticker) | set(bear_by_ticker))
    debates = []
    for ticker in all_tickers:
        debate: dict[str, Any] = {"ticker": ticker}
        if ticker in bull_by_ticker:
            debate["bull"] = bull_by_ticker[ticker]
        if ticker in bear_by_ticker:
            debate["bear"] = bear_by_ticker[ticker]
        debates.append(debate)

    return {
        "date": state.get("current_date", ""),
        # Macro regime
        "macro": {
            "regime": macro.get("regime", "uncertain"),
            "sentiment_score": macro.get("sentiment_score", 0.0),
            "key_drivers": macro.get("key_drivers", []),
            "sector_tilts": macro.get("sector_tilts", []),
            "risks": macro.get("risks", []),
        },
        # Debates per ticker (bull + bear arguments)
        "debates": debates,
        # Layer 1 summaries
        "l1_summary": {
            "social": social.get("summary", ""),
            "news": news.get("summary", ""),
        },
        # Supporting context
        "tickers_analyzed": [c["ticker"] for c in valid_companies if "ticker" in c],
        "company_analyses": valid_companies,
        "price_analyses": valid_prices,
        "macro_themes": social.get("macro_themes", []) + news.get("macro_themes", []),
        "ticker_mentions": {
            "social": social.get("ticker_mentions", []),
            "news_clusters": news.get("story_clusters", []),
        },
        "messages_analyzed": news.get("messages_analyzed", 0),
        # Pipeline errors (for downstream visibility)
        "errors": {
            "social_failed": social.get("error", False),
            "news_failed": news.get("error", False),
            "company_failures": [
                c["ticker"] for c in companies if c.get("error") and "ticker" in c
            ],
            "price_failures": [p["ticker"] for p in prices if p.get("error") and "ticker" in p],
            "bull_failures": [
                item["ticker"] for item in bull_analyses if item.get("error") and "ticker" in item
            ],
            "bear_failures": [
                item["ticker"] for item in bear_analyses if item.get("error") and "ticker" in item
            ],
            "macro_failed": macro.get("error", False),
        },
    }
