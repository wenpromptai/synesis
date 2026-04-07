"""Brief compiler — deterministic assembly of pipeline outputs.

No LLM. Collects all agent outputs from state and assembles them into
a structured brief. Expanded in Phase 3C with debate results and ranking.
"""

from __future__ import annotations

from typing import Any


def compile_brief(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble an intelligence brief from pipeline state.

    Args:
        state: The full IntelligenceState dict.

    Returns:
        Structured brief dict with macro regime, trade ideas,
        quick takes, company analyses, and macro themes.
    """
    social = state.get("social_analysis", {})
    news = state.get("news_analysis", {})
    companies = state.get("company_analyses", [])
    valid_companies = [c for c in companies if not c.get("error")]
    macro = state.get("macro_view", {})
    equity = state.get("equity_ideas", {})

    # Split trade ideas by conviction (abs sentiment_score)
    all_ideas = equity.get("trade_ideas", [])
    trade_ideas = sorted(all_ideas, key=lambda i: abs(i.get("sentiment_score", 0)), reverse=True)
    high_conviction = [i for i in trade_ideas if abs(i.get("sentiment_score", 0)) >= 0.7]
    quick_takes = [i for i in trade_ideas if abs(i.get("sentiment_score", 0)) < 0.7]

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
        # Trade ideas (ranked by abs sentiment_score)
        "trade_ideas": high_conviction,
        "quick_takes": quick_takes,
        # Tier 1 summaries
        "tier1_summary": {
            "social": social.get("summary", ""),
            "news": news.get("summary", ""),
        },
        # Supporting context
        "tickers_analyzed": [c["ticker"] for c in valid_companies if "ticker" in c],
        "company_analyses": valid_companies,
        "macro_themes": social.get("macro_themes", []) + news.get("macro_themes", []),
        "ticker_mentions": {
            "social": social.get("ticker_mentions", []),
            "news_clusters": news.get("story_clusters", []),
        },
        "messages_analyzed": news.get("messages_analyzed", 0),
    }
