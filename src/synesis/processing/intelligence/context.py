"""Context formatters for the intelligence pipeline.

Per-ticker formatters filter pipeline state to a single ticker's context
for use in the per-ticker bull/bear debate fan-out. The macro formatter
provides regime context for the Trader node (Phase 3D).
"""

from __future__ import annotations

from typing import Any


def format_macro_context(state: dict[str, Any]) -> str:
    """Format MacroView for an LLM prompt."""
    macro = state.get("macro_view", {})
    if not macro:
        return "## Macro Context\nNo macro assessment available."

    lines = ["## Macro Context"]
    lines.append(
        f"- **Regime**: {macro.get('regime', '?')} "
        f"(sentiment: {macro.get('sentiment_score', 0):+.1f})"
    )
    for driver in macro.get("key_drivers", []):
        lines.append(f"- Driver: {driver}")
    for tilt in macro.get("thematic_tilts", []):
        etf = tilt.get("etf")
        etf_str = f" [{etf}]" if etf else ""
        lines.append(
            f"- Thematic tilt: {tilt.get('theme', '?')}{etf_str} "
            f"({tilt.get('sentiment_score', 0):+.1f}) — {tilt.get('reasoning', '')}"
        )
    for risk in macro.get("risks", []):
        lines.append(f"- Risk: {risk}")
    return "\n".join(lines)


def _format_single_company(c: dict[str, Any]) -> list[str]:
    """Format a single company analysis dict into lines."""
    lines: list[str] = []
    ticker = c.get("ticker", "?")
    lines.append(f"\n### {ticker} ({c.get('company_name', '')})")
    if c.get("sector"):
        lines.append(f"Sector: {c['sector']} / {c.get('industry', '')}")

    # ── Financial Health ──
    health = c.get("financial_health", {})
    if health:
        # Computed scores
        score_parts = []
        if health.get("piotroski_f") is not None:
            score_parts.append(f"Piotroski F-Score={health['piotroski_f']}/9")
        if health.get("beneish_m") is not None:
            score_parts.append(f"Beneish M-Score={health['beneish_m']:.2f}")
        if score_parts:
            lines.append(f"Scores: {', '.join(score_parts)}")

        # Valuation
        val_parts = []
        if health.get("market_cap") is not None:
            lines.append(f"Market Cap: ${health['market_cap'] / 1e9:.1f}B")
        if health.get("price_to_book") is not None:
            val_parts.append(f"P/B={health['price_to_book']:.1f}")
        if health.get("ev_to_ebitda") is not None:
            val_parts.append(f"EV/EBITDA={health['ev_to_ebitda']:.1f}")
        if health.get("forward_eps") is not None:
            val_parts.append(f"Fwd EPS=${health['forward_eps']:.2f}")
        if val_parts:
            lines.append(f"Valuation: {', '.join(val_parts)}")

        # Profitability
        prof_parts = []
        if health.get("roe") is not None:
            prof_parts.append(f"ROE={health['roe']:.1%}")
        if health.get("roa") is not None:
            prof_parts.append(f"ROA={health['roa']:.1%}")
        if health.get("gross_margin") is not None:
            prof_parts.append(f"Gross={health['gross_margin']:.1%}")
        if health.get("operating_margin") is not None:
            prof_parts.append(f"Operating={health['operating_margin']:.1%}")
        if health.get("profit_margin") is not None:
            prof_parts.append(f"Net={health['profit_margin']:.1%}")
        if prof_parts:
            lines.append(f"Margins: {', '.join(prof_parts)}")

        # Growth
        if health.get("revenue_growth") is not None:
            lines.append(f"Revenue Growth: {health['revenue_growth']:.1%}")

        # Balance sheet
        bs_parts = []
        if health.get("debt_to_equity") is not None:
            bs_parts.append(f"D/E={health['debt_to_equity']:.2f}")
        if health.get("current_ratio") is not None:
            bs_parts.append(f"Current={health['current_ratio']:.2f}")
        if health.get("quick_ratio") is not None:
            bs_parts.append(f"Quick={health['quick_ratio']:.2f}")
        if bs_parts:
            lines.append(f"Balance Sheet: {', '.join(bs_parts)}")

        cf_parts = []
        if health.get("free_cash_flow") is not None:
            cf_parts.append(f"FCF=${health['free_cash_flow'] / 1e9:.2f}B")
        if health.get("ebitda") is not None:
            cf_parts.append(f"EBITDA=${health['ebitda'] / 1e9:.2f}B")
        if health.get("total_cash") is not None:
            cf_parts.append(f"Cash=${health['total_cash'] / 1e9:.2f}B")
        if health.get("total_debt") is not None:
            cf_parts.append(f"Debt=${health['total_debt'] / 1e9:.2f}B")
        if cf_parts:
            lines.append(f"Cash Flow & Capital: {', '.join(cf_parts)}")

        # Other
        if health.get("beta") is not None:
            lines.append(f"Beta: {health['beta']:.2f}")
        if health.get("short_percent_of_float") is not None:
            lines.append(f"Short Interest: {health['short_percent_of_float']:.1%} of float")

        # Quarterly trends
        if health.get("quarterly_revenue_trend"):
            trend = health["quarterly_revenue_trend"]
            trend_str = ", ".join(
                f"{q.get('period', '?')}: ${q.get('value', 0) / 1e9:.2f}B"
                for q in trend
                if q.get("value") is not None
            )
            if trend_str:
                lines.append(f"Revenue Trend: {trend_str}")

        if health.get("quarterly_eps_trend"):
            trend = health["quarterly_eps_trend"]
            trend_str = ", ".join(
                f"{q.get('period', '?')}: ${q.get('value', 0):.2f}"
                for q in trend
                if q.get("value") is not None
            )
            if trend_str:
                lines.append(f"EPS Trend: {trend_str}")

    # ── Insider Activity ──
    insider = c.get("insider_signal", {})
    if insider:
        mspr = insider.get("mspr")
        if mspr is not None:
            lines.append(
                f"Insiders: MSPR={mspr:+.2f}, "
                f"buys={insider.get('buy_count', 0)} (${insider.get('total_buy_value', 0) / 1e6:.1f}M), "
                f"sells={insider.get('sell_count', 0)} (${insider.get('total_sell_value', 0) / 1e6:.1f}M), "
                f"cluster={'YES' if insider.get('cluster_detected') else 'no'}"
            )
        if insider.get("form144_count"):
            lines.append(f"Form 144 filings: {insider['form144_count']}")
        if insider.get("csuite_activity"):
            lines.append(f"C-suite: {insider['csuite_activity']}")
        for txn in insider.get("notable_transactions", []):
            lines.append(f"  - {txn}")

    # ── Analyst Consensus ──
    analyst = c.get("analyst_consensus", {})
    if analyst:
        buy = analyst.get("buy_count", 0)
        hold = analyst.get("hold_count", 0)
        sell = analyst.get("sell_count", 0)
        total = buy + hold + sell
        if total:
            lines.append(f"Analysts: Buy={buy}, Hold={hold}, Sell={sell}")
        pt_mean = analyst.get("price_target_mean")
        pt_low = analyst.get("price_target_low")
        pt_high = analyst.get("price_target_high")
        current = analyst.get("current_price")
        if pt_mean and current and current > 0:
            upside = (pt_mean - current) / current * 100
            range_str = ""
            if pt_low is not None and pt_high is not None:
                range_str = f", range ${pt_low:.0f}-${pt_high:.0f}"
            lines.append(
                f"Price Targets: mean=${pt_mean:.0f} ({upside:+.0f}%){range_str}, "
                f"current=${current:.2f}"
            )
        for action in analyst.get("recent_actions", []):
            lines.append(f"  - {action}")

    # ── Red Flags ──
    for rf in c.get("red_flags", []):
        lines.append(f"[{rf.get('severity', '?')}] {rf.get('flag', '')}: {rf.get('evidence', '')}")

    # ── Qualitative Insights (from 10-K/10-Q/8-K) ──
    if c.get("business_summary"):
        lines.append(f"Business: {c['business_summary']}")
    if c.get("earnings_quality"):
        lines.append(f"Earnings Quality: {c['earnings_quality']}")
    if c.get("risk_assessment"):
        lines.append(f"Risk Assessment: {c['risk_assessment']}")
    if c.get("geographic_exposure"):
        lines.append(f"Geographic Exposure: {c['geographic_exposure']}")
    if c.get("key_customers_suppliers"):
        lines.append(f"Key Customers/Suppliers: {c['key_customers_suppliers']}")
    if c.get("forward_outlook"):
        lines.append(f"Forward Outlook: {c['forward_outlook']}")
    if c.get("competitive_position"):
        lines.append(f"Competitive Position: {c['competitive_position']}")

    # ── Cross-Referenced Insights ──
    if c.get("insider_vs_financials"):
        lines.append(f"Insider vs Financials: {c['insider_vs_financials']}")
    if c.get("disclosure_consistency"):
        lines.append(f"Disclosure Consistency: {c['disclosure_consistency']}")

    # ── Key Findings ──
    if c.get("primary_thesis"):
        lines.append(f"Primary Thesis: {c['primary_thesis']}")
    risks = c.get("key_risks", [])
    if risks:
        lines.append("Key Risks: " + "; ".join(risks))
    triggers = c.get("monitoring_triggers", [])
    if triggers:
        lines.append("Monitoring Triggers: " + "; ".join(triggers))

    return lines


def _format_single_price(p: dict[str, Any]) -> list[str]:
    """Format a single price analysis dict into lines."""
    lines: list[str] = []
    ticker = p.get("ticker", "?")
    lines.append(f"\n### {ticker}")

    # Spot price
    if p.get("spot_price"):
        change = p.get("change_1d_pct")
        change_str = f" ({change:+.1f}%)" if change is not None else ""
        lines.append(f"Price: ${p['spot_price']:.2f}{change_str}")

    # Technical indicators
    tech_parts = []
    if p.get("ema_8") is not None and p.get("ema_21") is not None:
        tech_parts.append(f"EMA 8/21: ${p['ema_8']:.2f} / ${p['ema_21']:.2f}")
        if p.get("ema_cross"):
            tech_parts.append(f"EMA cross: {p['ema_cross']}")
    if p.get("rsi_14") is not None:
        tech_parts.append(f"RSI-14: {p['rsi_14']:.1f}")
    if p.get("adx") is not None:
        tech_parts.append(f"ADX: {p['adx']:.1f}")
    if p.get("macd_histogram") is not None:
        tech_parts.append(f"MACD hist: {p['macd_histogram']:.3f}")
        if p.get("macd_signal_cross"):
            tech_parts.append(f"MACD cross: {p['macd_signal_cross']}")
    if p.get("atr_percent") is not None:
        tech_parts.append(f"ATR%: {p['atr_percent']:.2f}%")
    if tech_parts:
        lines.append(f"Technicals: {', '.join(tech_parts)}")

    # Bollinger / z-score
    bb_parts = []
    if p.get("bb_width_percentile") is not None:
        bb_parts.append(f"BB width %ile: {p['bb_width_percentile']:.0f}")
    if p.get("bb_percent_b") is not None:
        bb_parts.append(f"%B: {p['bb_percent_b']:.2f}")
    if p.get("price_zscore") is not None:
        bb_parts.append(f"Z-score: {p['price_zscore']:.2f}")
    if bb_parts:
        lines.append(f"Bollinger: {', '.join(bb_parts)}")

    # Volume
    vol_parts = []
    if p.get("volume_ratio") is not None:
        vol_parts.append(f"Vol ratio: {p['volume_ratio']:.2f}x avg")
    if p.get("obv_trend"):
        vol_parts.append(f"OBV: {p['obv_trend']}")
    if vol_parts:
        lines.append(f"Volume: {', '.join(vol_parts)}")

    # Support/Resistance
    if p.get("nearest_support") is not None or p.get("nearest_resistance") is not None:
        sr_parts = []
        if p.get("nearest_support") is not None:
            sr_parts.append(f"Support: ${p['nearest_support']:.2f}")
        if p.get("nearest_resistance") is not None:
            sr_parts.append(f"Resistance: ${p['nearest_resistance']:.2f}")
        lines.append(f"Levels: {', '.join(sr_parts)}")

    # Options metrics
    opt_parts = []
    if p.get("atm_iv") is not None:
        opt_parts.append(f"ATM IV: {p['atm_iv']:.1%}")
    if p.get("realized_vol_30d") is not None:
        opt_parts.append(f"RV-30d: {p['realized_vol_30d']:.1%}")
    if p.get("iv_rv_spread") is not None:
        opt_parts.append(f"IV-RV spread: {p['iv_rv_spread']:+.1%}")
    if p.get("put_call_volume_ratio") is not None:
        opt_parts.append(f"P/C ratio: {p['put_call_volume_ratio']:.2f}")
    if p.get("atm_skew_ratio") is not None:
        opt_parts.append(f"Skew: {p['atm_skew_ratio']:.2f}")
    if p.get("days_to_expiry") is not None:
        opt_parts.append(f"DTE: {p['days_to_expiry']}")
    if opt_parts:
        lines.append(f"Options: {', '.join(opt_parts)}")

    # Notable setups
    if p.get("notable_setups"):
        lines.append("Notable Setups: " + "; ".join(p["notable_setups"]))

    # LLM narratives
    if p.get("technical_narrative"):
        lines.append(f"Technical Narrative: {p['technical_narrative']}")
    if p.get("options_narrative"):
        lines.append(f"Options Narrative: {p['options_narrative']}")

    return lines


# =============================================================================
# Debate History Formatter (used by bull/bear researchers in multi-round debate)
# =============================================================================


def format_debate_history(history: list[dict[str, Any]]) -> str:
    """Format prior debate arguments for an LLM prompt."""
    if not history:
        return ""
    lines = ["## Prior Debate"]
    for arg in history:
        if arg.get("error"):
            continue
        role = arg.get("role", "?").upper()
        lines.append(f"\n### {role} (Round {arg.get('round', '?')})")
        lines.append(arg.get("argument", ""))
        evidence = arg.get("key_evidence", [])
        if evidence:
            lines.append("Key evidence:")
            for e in evidence:
                lines.append(f"- {e}")
    return "\n".join(lines)


def format_debate_summary_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format debate for the Trader — last round's full argument + key_evidence per side.

    Only the final round for each side is included (the most refined take).
    Earlier rounds are discarded to keep context manageable.
    """
    bull_analyses = state.get("bull_analyses", [])
    bear_analyses = state.get("bear_analyses", [])

    bull_items = sorted(
        [b for b in bull_analyses if b.get("ticker") == ticker and not b.get("error")],
        key=lambda x: x.get("round", 0),
    )
    bear_items = sorted(
        [b for b in bear_analyses if b.get("ticker") == ticker and not b.get("error")],
        key=lambda x: x.get("round", 0),
    )

    if not bull_items and not bear_items:
        return f"## Debate: {ticker}\n[No debate available — analysis failed or missing]"

    lines = [f"## Debate: {ticker}"]

    if bull_items:
        last_bull = bull_items[-1]
        lines.append(f"\n### BULL (Round {last_bull.get('round', '?')})")
        if last_bull.get("variant_vs_consensus"):
            lines.append(f"**Variant:** {last_bull['variant_vs_consensus']}")
        if last_bull.get("estimated_upside_downside"):
            lines.append(f"**Target:** {last_bull['estimated_upside_downside']}")
        lines.append(last_bull.get("argument") or "[argument not available]")
        for e in last_bull.get("key_evidence", []):
            lines.append(f"- {e}")
        if last_bull.get("catalyst"):
            lines.append(
                f"**Catalyst:** {last_bull['catalyst']} ({last_bull.get('catalyst_timeline', '')})"
            )
        if last_bull.get("what_would_change_my_mind"):
            lines.append(f"**Invalidation:** {last_bull['what_would_change_my_mind']}")

    if bear_items:
        last_bear = bear_items[-1]
        lines.append(f"\n### BEAR (Round {last_bear.get('round', '?')})")
        if last_bear.get("variant_vs_consensus"):
            lines.append(f"**Variant:** {last_bear['variant_vs_consensus']}")
        if last_bear.get("estimated_upside_downside"):
            lines.append(f"**Target:** {last_bear['estimated_upside_downside']}")
        lines.append(last_bear.get("argument") or "[argument not available]")
        for e in last_bear.get("key_evidence", []):
            lines.append(f"- {e}")
        if last_bear.get("catalyst"):
            lines.append(
                f"**Catalyst:** {last_bear['catalyst']} ({last_bear.get('catalyst_timeline', '')})"
            )
        if last_bear.get("what_would_change_my_mind"):
            lines.append(f"**Invalidation:** {last_bear['what_would_change_my_mind']}")

    return "\n".join(lines)


# =============================================================================
# Per-Ticker Context Filters (used by per-ticker bull/bear debate fan-out)
# =============================================================================


def format_social_context_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format social sentiment filtered to a single ticker."""
    social = state.get("social_analysis", {})
    if not social:
        return "## Social Sentiment\nNo social analysis available."

    mentions = [m for m in social.get("ticker_mentions", []) if m.get("ticker") == ticker]
    if not mentions:
        return f"## Social Sentiment\nNo social mentions for {ticker}."

    lines = [f"## Social Sentiment for {ticker}"]
    for mention in mentions:
        accounts = ", ".join(mention.get("source_accounts", []))
        lines.append(f"- {mention.get('context', '')} [from: {accounts}]")
    return "\n".join(lines)


def format_news_context_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format news analysis filtered to clusters mentioning a single ticker."""
    news = state.get("news_analysis", {})
    if not news:
        return "## News\nNo news analysis available."

    relevant_clusters = []
    for cluster in news.get("story_clusters", []):
        cluster_tickers = {t.get("ticker") for t in cluster.get("tickers", [])}
        if ticker in cluster_tickers:
            relevant_clusters.append(cluster)

    if not relevant_clusters:
        return f"## News\nNo news clusters for {ticker}."

    lines = [f"## News for {ticker}"]
    for cluster in relevant_clusters:
        urgency = cluster.get("urgency", "normal")
        event_type = cluster.get("event_type", "other")
        lines.append(f"\n### {cluster.get('headline', '?')} [{event_type}, urgency={urgency}]")
        for fact in cluster.get("key_facts", []):
            lines.append(f"- {fact}")
        for t in cluster.get("tickers", []):
            lines.append(f"- Ticker: **{t.get('ticker', '?')}** — {t.get('context', '')}")
    return "\n".join(lines)


def format_consensus_context_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format consensus expectations for debate agents to argue against.

    Pulls from CompanyAnalysis (analyst consensus, financial health) and
    PriceAnalysis (spot price, vol regime) to build a concise baseline
    of what the market currently prices in.
    """
    companies = state.get("company_analyses", [])
    company = next(
        (c for c in companies if c.get("ticker") == ticker and not c.get("error")),
        None,
    )
    prices = state.get("price_analyses", [])
    price = next(
        (p for p in prices if p.get("ticker") == ticker and not p.get("error")),
        None,
    )

    if not company and not price:
        return f"## Consensus View\nNo consensus data available for {ticker}."

    lines = [f"## Consensus View for {ticker}"]
    lines.append(
        "This is what the market currently prices in. Your job is to argue "
        "where consensus is WRONG — not to repeat what it already knows."
    )

    if company:
        analyst = company.get("analyst_consensus", {})
        health = company.get("financial_health", {})

        # Analyst ratings
        buy = analyst.get("buy_count", 0)
        hold = analyst.get("hold_count", 0)
        sell = analyst.get("sell_count", 0)
        total = buy + hold + sell
        if total:
            lines.append(f"- Analyst consensus: {buy} Buy / {hold} Hold / {sell} Sell")

        # Price targets
        pt_mean = analyst.get("price_target_mean")
        pt_low = analyst.get("price_target_low")
        pt_high = analyst.get("price_target_high")
        current = analyst.get("current_price")
        if pt_mean and current and current > 0:
            upside = (pt_mean - current) / current * 100
            parts = [f"mean ${pt_mean:.0f} ({upside:+.0f}% implied)"]
            if pt_low is not None and pt_high is not None:
                parts.append(f"range ${pt_low:.0f}–${pt_high:.0f}")
            lines.append(f"- Price targets: {', '.join(parts)}")
            lines.append(f"- Current price: ${current:.2f}")

        # Key financial expectations
        fwd_parts = []
        if health.get("forward_eps") is not None:
            fwd_parts.append(f"Fwd EPS ${health['forward_eps']:.2f}")
        if health.get("ev_to_ebitda") is not None:
            fwd_parts.append(f"EV/EBITDA {health['ev_to_ebitda']:.1f}x")
        if health.get("revenue_growth") is not None:
            fwd_parts.append(f"Rev growth {health['revenue_growth']:.1%}")
        if fwd_parts:
            lines.append(f"- Market expectations: {', '.join(fwd_parts)}")

        # Positioning signals
        if health.get("short_percent_of_float") is not None:
            lines.append(f"- Short interest: {health['short_percent_of_float']:.1%} of float")

        # Recent analyst actions
        for action in analyst.get("recent_actions", [])[:3]:
            lines.append(f"  - {action}")

    if price:
        # Vol regime context
        iv = price.get("atm_iv")
        rv = price.get("realized_vol_30d")
        spread = price.get("iv_rv_spread")
        if iv is not None and rv is not None:
            regime = "cheap" if (spread or 0) < 0 else "expensive"
            lines.append(
                f"- Vol regime: ATM IV {iv:.1%} vs 30d RV {rv:.1%} "
                f"(IV is {regime} relative to realized)"
            )

    return "\n".join(lines)


def format_company_context_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format company analysis for a single ticker."""
    companies = state.get("company_analyses", [])
    match = next(
        (c for c in companies if c.get("ticker") == ticker and not c.get("error")),
        None,
    )
    if not match:
        return f"## Company Analysis\nNo company analysis available for {ticker}."

    lines = ["## Company Analysis"]
    lines.extend(_format_single_company(match))
    return "\n".join(lines)


def format_screener_context_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format the screener's thematic angle for a ticker.

    Gives debate agents the CIO's thesis seed — why this ticker was selected
    and what angle to investigate.
    """
    ctx = state.get("screener_context", {})
    for pick in ctx.get("selected", []):
        if pick.get("ticker") == ticker:
            lines = ["## Screener Thesis Seed"]
            lines.append(f"**Thematic angle:** {pick.get('thematic_angle', 'N/A')}")
            lines.append(f"**Direction lean:** {pick.get('direction_lean', 'N/A')}")
            lines.append(f"**Signal strength:** {pick.get('signal_strength', 'N/A')}")
            if pick.get("research_note"):
                lines.append(f"**Research note:** {pick['research_note']}")
            if pick.get("is_wildcard"):
                lines.append(
                    "*This ticker was added as a wildcard — not in the original signal pool.*"
                )
            return "\n".join(lines)
    return ""


def format_price_context_for_ticker(state: dict[str, Any], ticker: str) -> str:
    """Format price analysis for a single ticker."""
    prices = state.get("price_analyses", [])
    match = next(
        (p for p in prices if p.get("ticker") == ticker and not p.get("error")),
        None,
    )
    if not match:
        return f"## Price Analysis\nNo price analysis available for {ticker}."

    lines = ["## Price Analysis"]
    lines.extend(_format_single_price(match))
    return "\n".join(lines)
