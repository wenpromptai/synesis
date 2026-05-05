"""Brief compiler — deterministic assembly of pipeline outputs.

No LLM. Collects all agent outputs from state and assembles them into
a structured brief with debate arguments grouped by ticker.

Also provides `format_brief_as_markdown` for saving briefs to the
knowledge graph at `docs/kg/raw/synesis_briefs/`.
"""

from __future__ import annotations

import re
from typing import Any, cast

from synesis.core.logging import get_logger

logger = get_logger(__name__)

# OpenAI web search leaks internal reference tokens like
# "citeturn0search12turn0search0turn0search13" into responses.
_LLM_ARTIFACT_RE = re.compile(r"(?:cite)?turn\d+search\d+(?:turn\d+search\d+)*")


def _strip_llm_artifacts(text: str) -> str:
    """Strip broken OpenAI web search reference tokens from text."""
    return _LLM_ARTIFACT_RE.sub("", text).strip()


def _sanitize_brief(obj: Any) -> Any:
    """Recursively strip LLM artifacts from all strings in a brief dict."""
    if isinstance(obj, str):
        return _strip_llm_artifacts(obj)
    if isinstance(obj, list):
        return [_sanitize_brief(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_brief(v) for k, v in obj.items()}
    return obj


def _extract_macro_dict(macro: dict[str, Any]) -> dict[str, Any]:
    """Extract the macro regime section from a macro_view state dict."""
    return {
        "regime": macro.get("regime", "uncertain"),
        "sentiment_score": macro.get("sentiment_score", 0.0),
        "key_drivers": macro.get("key_drivers", []),
        "thematic_tilts": macro.get("thematic_tilts", []),
        "event_analysis": macro.get("event_analysis", ""),
        "positioning_signals": macro.get("positioning_signals", ""),
        "risks": macro.get("risks", []),
    }


def _extract_watchlist_dict(state: dict[str, Any], watchlist_ctx: dict[str, Any]) -> dict[str, Any]:
    """Extract the watchlist section from state + watchlist context."""
    return {
        "l1_tickers": state.get("l1_tickers", []),
        "selected": watchlist_ctx.get("selected", []),
        "themes": watchlist_ctx.get("themes", []),
        "dropped": watchlist_ctx.get("dropped", []),
        "drop_reasons": watchlist_ctx.get("drop_reasons", []),
    }


def _format_macro_regime_markdown(macro: dict[str, Any]) -> list[str]:
    """Format macro regime section as markdown lines."""
    regime = macro.get("regime", "unknown")
    sentiment = macro.get("sentiment_score", 0.0)
    lines: list[str] = []

    lines.append("## Macro Regime")
    lines.append(f"- **Regime:** {regime} (sentiment: {sentiment:+.1f})")
    for driver in macro.get("key_drivers", []):
        lines.append(f"- **Driver:** {driver}")

    event_analysis = macro.get("event_analysis", "")
    if event_analysis:
        lines.append(f"\n### Event Analysis\n{event_analysis}")

    positioning = macro.get("positioning_signals", "")
    if positioning:
        lines.append(f"\n### Positioning Signals\n{positioning}")

    tilts = macro.get("thematic_tilts", [])
    if tilts:
        lines.append("\n### Thematic Tilts")
        for tilt in tilts:
            theme = tilt.get("theme", "?")
            tilt_score = tilt.get("sentiment_score", 0.0)
            reasoning = tilt.get("reasoning", "")
            etf = tilt.get("etf")
            etf_str = f" [{etf}]" if etf else ""
            persistence = tilt.get("persistence", "")
            pers_str = f" ({persistence})" if persistence else ""
            lines.append(f"- **{theme}**{etf_str}{pers_str} ({tilt_score:+.1f}) — {reasoning}")
            for evidence in tilt.get("key_evidence", []):
                lines.append(f"  - {evidence}")
            related = tilt.get("related_tickers", [])
            if related:
                lines.append(f"  - Tickers: {', '.join(related)}")
            catalyst = tilt.get("catalyst_date", "")
            if catalyst:
                lines.append(f"  - Catalyst: {catalyst}")

    for risk in macro.get("risks", []):
        lines.append(f"- **Risk:** {risk}")
    lines.append("")

    return lines


def _format_watchlist_markdown(watchlist: dict[str, Any]) -> list[str]:
    """Format watchlist section as markdown lines."""
    l1_pool = watchlist.get("l1_tickers", [])
    picks = watchlist.get("selected", [])
    if not l1_pool and not picks:
        return []

    lines: list[str] = []
    lines.append("## Watchlist")
    lines.append(f"**Signal pool:** {len(l1_pool)} tickers from Layer 1")
    lines.append(f"**Selected:** {len(picks)} for deep analysis")
    themes = watchlist.get("themes", [])
    if themes:
        lines.append(f"**Themes:** {', '.join(themes)}")
    lines.append("")
    if picks:
        lines.append("### Selected")
        for pick in picks:
            direction = pick.get("direction_lean", "?")
            wildcard = " (wildcard)" if pick.get("is_wildcard") else ""
            lines.append(
                f"- **{pick.get('ticker', '?')}** ({direction}{wildcard}) — "
                f"{pick.get('thematic_angle', '')}"
            )
            if pick.get("research_note"):
                lines.append(f"  {pick['research_note']}")
        lines.append("")
    dropped = watchlist.get("dropped", [])
    drop_reasons = watchlist.get("drop_reasons", [])
    if dropped:
        lines.append("### Dropped")
        for i, ticker in enumerate(dropped):
            reason = drop_reasons[i] if i < len(drop_reasons) else ""
            lines.append(f"- {ticker} — {reason}")
        lines.append("")

    return lines


def compile_brief(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble an intelligence brief from pipeline state.

    Args:
        state: The full AnalyzeState dict.

    Returns:
        Structured brief dict with macro regime, debates per ticker,
        company analyses, price analyses, and macro themes.
    """
    social = state.get("social_analysis", {})
    news = state.get("news_analysis", {})
    ticker_research = state.get("ticker_research", {})
    companies = state.get("company_analyses", [])
    valid_companies = [c for c in companies if not c.get("error")]
    prices = state.get("price_analyses", [])
    valid_prices = [p for p in prices if not p.get("error")]
    macro = state.get("macro_view", {})
    watchlist_ctx = state.get("watchlist_context", {})
    trade_ideas_raw = state.get("trade_ideas", [])
    valid_trade_ideas = [t for t in trade_ideas_raw if not t.get("error")]
    portfolio_note = state.get("portfolio_note", "")

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

    # Log compilation summary
    error_trade_ideas = [t for t in trade_ideas_raw if t.get("error")]
    logger.info(
        "Brief compiled",
        tickers_analyzed=len(valid_companies),
        debates=len(debates),
        trade_ideas=len(valid_trade_ideas),
        trade_idea_errors=len(error_trade_ideas),
        has_portfolio_note=bool(portfolio_note),
        company_failures=[c["ticker"] for c in companies if c.get("error") and "ticker" in c],
        price_failures=[p["ticker"] for p in prices if p.get("error") and "ticker" in p],
    )
    if not valid_trade_ideas:
        logger.warning(
            "Brief has 0 trade ideas",
            raw_trade_ideas=len(trade_ideas_raw),
            errored_trade_ideas=len(error_trade_ideas),
            portfolio_note_preview=portfolio_note[:200] if portfolio_note else "(empty)",
        )

    return cast(
        dict[str, Any],
        _sanitize_brief(
            {
                "date": state.get("current_date", ""),
                "macro": _extract_macro_dict(macro) if macro and not macro.get("error") else {},
                "watchlist": _extract_watchlist_dict(state, watchlist_ctx) if watchlist_ctx else {},
                # Debates per ticker (bull + bear arguments)
                "debates": debates,
                # Layer 1 summaries (empty when not run, e.g. analyze path)
                "l1_summary": {
                    "social": social.get("summary", ""),
                    "news": news.get("summary", ""),
                },
                "social_research_context": social.get("research_context", []),
                "social_discovered_themes": social.get("discovered_themes", []),
                # Supporting context
                "tickers_analyzed": [c["ticker"] for c in valid_companies if "ticker" in c],
                "ticker_research": ticker_research.get("research", [])
                if not ticker_research.get("error")
                else [],
                "company_analyses": valid_companies,
                "price_analyses": valid_prices,
                "macro_themes": social.get("macro_themes", []) + news.get("macro_themes", []),
                "ticker_mentions": {
                    "social": social.get("ticker_mentions", []),
                    "news_clusters": news.get("story_clusters", []),
                },
                "messages_analyzed": news.get("messages_analyzed", 0),
                # Trade ideas from Trader
                "trade_ideas": valid_trade_ideas,
                "portfolio_note": portfolio_note,
                # Pipeline errors (for downstream visibility)
                "errors": {
                    "social_failed": social.get("error", False),
                    "news_failed": news.get("error", False),
                    "company_failures": [
                        c["ticker"] for c in companies if c.get("error") and "ticker" in c
                    ],
                    "price_failures": [
                        p["ticker"] for p in prices if p.get("error") and "ticker" in p
                    ],
                    "bull_failures": [
                        item["ticker"]
                        for item in bull_analyses
                        if item.get("error") and "ticker" in item
                    ],
                    "bear_failures": [
                        item["ticker"]
                        for item in bear_analyses
                        if item.get("error") and "ticker" in item
                    ],
                    "macro_failed": macro.get("error", False),
                    "ticker_research_failed": ticker_research.get("error", False),
                    "trader_failures": [
                        t
                        for item in trade_ideas_raw
                        if item.get("error") and "tickers" in item
                        for t in item["tickers"]
                    ],
                },
            }
        ),
    )


def format_brief_as_markdown(brief: dict[str, Any]) -> str:
    """Format a compiled brief as readable markdown for the knowledge graph.

    Saved to ``docs/kg/raw/synesis_briefs/YYYY-MM-DD.md`` by the pipeline job.
    Designed for future LLM compilation into KG nodes via ``/kg-compile``.
    """
    date = brief.get("date", "unknown")
    tickers = brief.get("tickers_analyzed", [])
    macro = brief.get("macro", {})
    regime = macro.get("regime", "unknown")
    sentiment = macro.get("sentiment_score", 0.0)
    trade_ideas = brief.get("trade_ideas", [])
    errors = brief.get("errors", {})

    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"date: {date}")
    lines.append(f"tickers: [{', '.join(tickers)}]")
    if macro:
        lines.append(f"regime: {regime}")
        lines.append(f"regime_sentiment: {sentiment}")
    lines.append(f"trade_count: {len(trade_ideas)}")
    has_errors = any(errors.get(k) for k in errors)
    lines.append(f"pipeline_errors: {'true' if has_errors else 'false'}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Intelligence Brief — {date}")
    lines.append("")

    # Macro (skip if not populated, e.g. analyze-only path)
    if macro:
        lines.extend(_format_macro_regime_markdown(macro))

    # Watchlist
    lines.extend(_format_watchlist_markdown(brief.get("watchlist", {})))

    if trade_ideas:
        lines.append("## Trade Ideas")
        lines.append("")
        portfolio_note_md = brief.get("portfolio_note", "")
        if portfolio_note_md:
            lines.append(portfolio_note_md)
            lines.append("")
        for idea in trade_ideas:
            idea_tickers = ", ".join(idea.get("tickers", []))
            structure = idea.get("trade_structure", "")
            lines.append(f"### {idea_tickers} — {structure}")
            if idea.get("thesis"):
                lines.append(f"- **Thesis:** {idea['thesis']}")
            # R/R framework
            entry = idea.get("entry_price")
            target = idea.get("target_price")
            stop = idea.get("stop_price")
            rr = idea.get("risk_reward_ratio")
            if entry is not None and target is not None and stop is not None:
                rr_str = f" (R/R {rr:.1f}:1)" if rr is not None else ""
                lines.append(
                    f"- **Entry:** ${entry:.2f} | **Target:** ${target:.2f} "
                    f"| **Stop:** ${stop:.2f}{rr_str}"
                )
            # Conviction
            tier = idea.get("conviction_tier")
            rationale = idea.get("conviction_rationale", "")
            if tier is not None:
                lines.append(f"- **Conviction:** Tier {tier} — {rationale}")
            catalyst = idea.get("catalyst", "")
            timeframe = idea.get("timeframe", "")
            if catalyst:
                tf_str = f" ({timeframe})" if timeframe else ""
                lines.append(f"- **Catalyst:** {catalyst}{tf_str}")
            elif timeframe:
                lines.append(f"- **Timeframe:** {timeframe}")
            if idea.get("key_risk"):
                lines.append(f"- **Key Risk:** {idea['key_risk']}")
            if idea.get("downside_scenario"):
                lines.append(f"- **Downside Scenario:** {idea['downside_scenario']}")
            if idea.get("expression_note"):
                lines.append(f"- **Vol Context:** {idea['expression_note']}")
            lines.append("")

    # Debates
    debates = brief.get("debates", [])
    if debates:
        lines.append("## Debates")
        lines.append("")
        for debate in debates:
            ticker = debate.get("ticker", "?")
            lines.append(f"### {ticker}")
            for side_key, label in [("bull", "Bull"), ("bear", "Bear")]:
                side = debate.get(side_key, {})
                if not side:
                    continue
                lines.append(f"\n**{label}:**")
                if side.get("variant_vs_consensus"):
                    lines.append(f"*Variant:* {side['variant_vs_consensus']}")
                if side.get("estimated_upside_downside"):
                    lines.append(f"*Target:* {side['estimated_upside_downside']}")
                lines.append(side.get("argument", "N/A"))
                if side.get("catalyst"):
                    timeline = side.get("catalyst_timeline", "")
                    tl_str = f" ({timeline})" if timeline else ""
                    lines.append(f"*Catalyst:* {side['catalyst']}{tl_str}")
                if side.get("what_would_change_my_mind"):
                    lines.append(f"*Invalidation:* {side['what_would_change_my_mind']}")
            lines.append("")

    # Company analyses
    company_analyses = brief.get("company_analyses", [])
    if company_analyses:
        lines.append("## Company Analyses")
        lines.append("")
        for ca in company_analyses:
            ticker = ca.get("ticker", "?")
            lines.append(f"### {ticker}")
            if ca.get("business_summary"):
                lines.append(f"- **Business:** {ca['business_summary']}")
            if ca.get("forward_outlook"):
                lines.append(f"- **Forward outlook:** {ca['forward_outlook']}")
            if ca.get("primary_thesis"):
                lines.append(f"- **Thesis:** {ca['primary_thesis']}")
            if ca.get("competitive_position"):
                lines.append(f"- **Competitive position:** {ca['competitive_position']}")
            if ca.get("key_customers_suppliers"):
                lines.append(f"- **Customers/Suppliers:** {ca['key_customers_suppliers']}")
            if ca.get("geographic_exposure"):
                lines.append(f"- **Geographic exposure:** {ca['geographic_exposure']}")
            if ca.get("earnings_quality"):
                lines.append(f"- **Earnings quality:** {ca['earnings_quality']}")
            if ca.get("risk_assessment"):
                lines.append(f"- **Risk:** {ca['risk_assessment']}")
            # Financial health
            fh = ca.get("financial_health", {})
            if fh.get("piotroski_f") is not None:
                lines.append(f"- **Piotroski F-Score:** {fh['piotroski_f']}/9")
            if fh.get("latest_filing_period"):
                lines.append(f"- **Filing period:** {fh['latest_filing_period']}")
            # Insider signal
            if ca.get("insider_signal"):
                ins = ca["insider_signal"]
                lines.append(
                    f"- **Insider signal:** MSPR {ins.get('mspr', 'N/A')}, "
                    f"buys {ins.get('buy_count', 0)}, sells {ins.get('sell_count', 0)}, "
                    f"cluster={'yes' if ins.get('cluster_detected') else 'no'}"
                )
            # Analyst consensus
            ac = ca.get("analyst_consensus", {})
            if ac.get("buy_count") or ac.get("hold_count") or ac.get("sell_count"):
                pt_str = ""
                if ac.get("price_target_mean"):
                    pt_str = f", PT mean=${ac['price_target_mean']:.0f}"
                lines.append(
                    f"- **Analyst consensus** ({ac.get('consensus_period', '?')}): "
                    f"Buy={ac.get('buy_count', 0)}, Hold={ac.get('hold_count', 0)}, "
                    f"Sell={ac.get('sell_count', 0)}{pt_str}"
                )
            if ca.get("red_flags"):
                for flag in ca["red_flags"]:
                    lines.append(f"- **Red flag:** {flag}")
            if ca.get("key_risks"):
                for risk in ca["key_risks"]:
                    lines.append(f"- **Key risk:** {risk}")
            if ca.get("insider_vs_financials"):
                lines.append(f"- **Insider vs financials:** {ca['insider_vs_financials']}")
            if ca.get("monitoring_triggers"):
                lines.append(f"- **Watch:** {'; '.join(ca['monitoring_triggers'])}")
            lines.append("")

    # Price analyses
    price_analyses = brief.get("price_analyses", [])
    if price_analyses:
        lines.append("## Price Analyses")
        lines.append("")
        for pa in price_analyses:
            ticker = pa.get("ticker", "?")
            lines.append(f"### {ticker}")
            if pa.get("spot_price") is not None:
                lines.append(f"- **Price:** ${pa['spot_price']:.2f}")
            if pa.get("change_1d_pct") is not None:
                lines.append(f"- **1d change:** {pa['change_1d_pct']:+.1f}%")
            if pa.get("rsi_14") is not None:
                lines.append(f"- **RSI(14):** {pa['rsi_14']:.1f}")
            if pa.get("adx") is not None:
                lines.append(f"- **ADX:** {pa['adx']:.1f}")
            if pa.get("ema_cross"):
                lines.append(f"- **EMA cross:** {pa['ema_cross']}")
            if pa.get("macd_signal_cross"):
                lines.append(f"- **MACD:** {pa['macd_signal_cross']}")
            if pa.get("atm_iv") is not None:
                lines.append(f"- **ATM IV:** {pa['atm_iv']:.1%}")
            if pa.get("realized_vol_30d") is not None:
                lines.append(f"- **RV(30d):** {pa['realized_vol_30d']:.1%}")
            if pa.get("iv_rv_spread") is not None:
                lines.append(f"- **IV-RV spread:** {pa['iv_rv_spread']:+.1%}")
            if pa.get("put_call_volume_ratio") is not None:
                lines.append(f"- **P/C ratio:** {pa['put_call_volume_ratio']:.2f}")
            if pa.get("notable_setups"):
                lines.append(f"- **Setups:** {', '.join(pa['notable_setups'])}")
            if pa.get("technical_narrative"):
                lines.append(f"- **Technical:** {pa['technical_narrative']}")
            if pa.get("options_narrative"):
                lines.append(f"- **Options:** {pa['options_narrative']}")
            lines.append("")

    # Signals
    l1 = brief.get("l1_summary", {})
    themes = brief.get("macro_themes", [])
    if l1 or themes:
        lines.append("## Signals")
        if l1.get("social"):
            lines.append(f"- **Social:** {l1['social']}")
        if l1.get("news"):
            lines.append(f"- **News:** {l1['news']}")
        for theme in themes:
            if isinstance(theme, dict):
                name = theme.get("theme", "?")
                ctx = theme.get("context", "")
                accounts = theme.get("source_accounts", [])
                acct_str = f" [from: {', '.join(accounts)}]" if accounts else ""
                lines.append(f"- **Macro theme:** {name} — {ctx}{acct_str}")
            else:
                lines.append(f"- **Macro theme:** {theme}")

        # Thematic research from social analysis
        research = brief.get("social_research_context", [])
        if research:
            lines.append("\n### Thematic Research")
            for item in research:
                lines.append(f"- {item}")
        discovered = brief.get("social_discovered_themes", [])
        if discovered:
            lines.append("\n### Discovered Themes")
            for item in discovered:
                lines.append(f"- {item}")
        lines.append("")

    # Pipeline health
    lines.append("## Pipeline Health")
    lines.append(f"- **Tickers analyzed:** {len(tickers)}")
    lines.append(f"- **Trade ideas:** {len(trade_ideas)}")
    if has_errors:
        for key, val in errors.items():
            if val:
                lines.append(f"- **{key}:** {val}")
    else:
        lines.append("- **Errors:** none")
    lines.append("")

    return "\n".join(lines)
