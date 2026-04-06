"""Brief compiler — deterministic assembly of pipeline outputs.

No LLM. Collects all agent outputs from state and assembles them into
a structured brief. Expanded in Phase 3C with debate results and ranking.
"""

from __future__ import annotations

from typing import Any


def compile_brief(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble a basic intelligence brief from pipeline state.

    Args:
        state: The full IntelligenceState dict.

    Returns:
        Structured brief dict with tier1 summaries, analyzed tickers,
        company analyses, and macro themes.
    """
    social = state.get("social_analysis", {})
    news = state.get("news_analysis", {})
    companies = state.get("company_analyses", [])
    valid_companies = [c for c in companies if not c.get("error")]

    return {
        "date": state.get("current_date", ""),
        "tier1_summary": {
            "social": social.get("summary", ""),
            "news": news.get("summary", ""),
        },
        "tickers_analyzed": [c["ticker"] for c in valid_companies if "ticker" in c],
        "company_analyses": valid_companies,
        "macro_themes": social.get("macro_themes", []) + news.get("macro_themes", []),
        "ticker_mentions": {
            "social": social.get("ticker_mentions", []),
            "news_clusters": news.get("story_clusters", []),
        },
        "messages_analyzed": news.get("messages_analyzed", 0),
    }
