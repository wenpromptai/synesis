"""Brief compiler — deterministic assembly of pipeline outputs.

No LLM. Collects all agent outputs from state and assembles them into
a structured brief with debate arguments grouped by ticker.

Also provides `format_brief_as_markdown` for saving briefs to the
knowledge graph at `docs/kg/raw/synesis_briefs/`.
"""

from __future__ import annotations

from typing import Any

from synesis.core.logging import get_logger

logger = get_logger(__name__)


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

    return {
        "date": state.get("current_date", ""),
        # Macro regime
        "macro": {
            "regime": macro.get("regime", "uncertain"),
            "sentiment_score": macro.get("sentiment_score", 0.0),
            "key_drivers": macro.get("key_drivers", []),
            "thematic_tilts": macro.get("thematic_tilts", []),
            "event_analysis": macro.get("event_analysis", ""),
            "positioning_signals": macro.get("positioning_signals", ""),
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
            "price_failures": [p["ticker"] for p in prices if p.get("error") and "ticker" in p],
            "bull_failures": [
                item["ticker"] for item in bull_analyses if item.get("error") and "ticker" in item
            ],
            "bear_failures": [
                item["ticker"] for item in bear_analyses if item.get("error") and "ticker" in item
            ],
            "macro_failed": macro.get("error", False),
            "trader_failures": [
                t
                for item in trade_ideas_raw
                if item.get("error") and "tickers" in item
                for t in item["tickers"]
            ],
        },
    }


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
    lines.append(f"regime: {regime}")
    lines.append(f"regime_sentiment: {sentiment}")
    lines.append(f"trade_count: {len(trade_ideas)}")
    has_errors = any(errors.get(k) for k in errors)
    lines.append(f"pipeline_errors: {'true' if has_errors else 'false'}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Intelligence Brief — {date}")
    lines.append("")

    # Macro
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
            lines.append(f"- **{theme}**{etf_str} ({tilt_score:+.1f}) — {reasoning}")

    for risk in macro.get("risks", []):
        lines.append(f"- **Risk:** {risk}")
    lines.append("")

    # Trade ideas
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
            if idea.get("catalyst"):
                lines.append(f"- **Catalyst:** {idea['catalyst']}")
            if idea.get("timeframe"):
                lines.append(f"- **Timeframe:** {idea['timeframe']}")
            if idea.get("key_risk"):
                lines.append(f"- **Key Risk:** {idea['key_risk']}")
            lines.append("")

    # Debates
    debates = brief.get("debates", [])
    if debates:
        lines.append("## Debates")
        lines.append("")
        for debate in debates:
            ticker = debate.get("ticker", "?")
            lines.append(f"### {ticker}")
            bull = debate.get("bull", {})
            bear = debate.get("bear", {})
            if bull:
                lines.append(f"**Bull:** {bull.get('argument', 'N/A')}")
            if bear:
                lines.append(f"**Bear:** {bear.get('argument', 'N/A')}")
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
            lines.append(f"- **Macro theme:** {theme}")
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
