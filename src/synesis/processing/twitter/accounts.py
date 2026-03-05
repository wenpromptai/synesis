"""Twitter account profiles for LLM context.

Each account has a category and description so the analyzer can
weight credibility, flag conflicts of interest, and cross-confirm
themes across different expertise areas.

To add a new account:
1. Add the handle to TWITTER_ACCOUNTS in .env
2. Add an entry to ACCOUNT_PROFILES below
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccountProfile:
    """Profile metadata for a Twitter account."""

    category: str
    description: str


# Case-insensitive lookup: keys are stored lowercase
ACCOUNT_PROFILES: dict[str, AccountProfile] = {
    "javierblas": AccountProfile(
        "macro_commodities",
        "Bloomberg commodities columnist. Oil, gas, metals, energy geopolitics.",
    ),
    "fuzzypandashort": AccountProfile(
        "short_seller",
        "Activist short-seller. Publishes fraud/accounting research reports. BIAS: always short.",
    ),
    "elerianm": AccountProfile(
        "macro",
        "Chief economic advisor at Allianz, ex-PIMCO CEO. Fed policy, global macro.",
    ),
    "lizannsonders": AccountProfile(
        "macro",
        "Schwab chief investment strategist. Macro cycles, sentiment indicators, breadth data.",
    ),
    "investingluc": AccountProfile(
        "technical",
        "Technical/momentum trader. Chart setups, SPY flow analysis.",
    ),
    "ewhispers": AccountProfile(
        "earnings",
        "Earnings estimates and whisper numbers. Pre/post-earnings data. Neutral data source.",
    ),
    "harmongreg": AccountProfile(
        "technical",
        "CMT/CFA, Dragonfly Capital. Chart patterns, multi-timeframe technical analysis.",
    ),
    "danzanger": AccountProfile(
        "technical",
        "Chart pattern trader. Breakout setups, volume analysis.",
    ),
    "hypertechinvest": AccountProfile(
        "sector_tech",
        "Deep-dive tech/AI/datacenter stock analysis. Fundamental + catalyst. BIAS: long tech/AI.",
    ),
    "michaeljburry": AccountProfile(
        "macro_contrarian",
        "Scion Capital founder (The Big Short). Contrarian macro, value. BIAS: perma-bear, cryptic posts.",
    ),
    "c_barraud": AccountProfile(
        "macro",
        "#1 ranked macro forecaster (Bloomberg). Economic data releases, global PMIs, surprise indices.",
    ),
    "spotgamma": AccountProfile(
        "options_flow",
        "Options analytics. Gamma exposure levels, dealer positioning, key strike levels, vol surface.",
    ),
    "kerrisdalecap": AccountProfile(
        "activist_fund",
        "Activist long/short hedge fund. Publishes research reports. BIAS: talks their book (long or short).",
    ),
    "biancoresearch": AccountProfile(
        "macro",
        "Jim Bianco. Rates, credit spreads, Fed policy, fixed income. Independent research.",
    ),
    "unusual_whales": AccountProfile(
        "options_flow",
        "Unusual options activity tracker. Congressional trades, dark pool prints. Neutral data source.",
    ),
    "bullflowio": AccountProfile(
        "options_flow",
        "Dark pool and options flow alerts. Large block trades, sweep detection.",
    ),
    "nolimitgains": AccountProfile(
        "geopolitical",
        "Geopolitical events and breaking market news aggregator. Fast but verify claims.",
    ),
    "muddywatersre": AccountProfile(
        "short_seller",
        "Muddy Waters Research (Carson Block). Fraud and accounting short reports. BIAS: always short.",
    ),
    "nicktimiraos": AccountProfile(
        "fed_reporter",
        "WSJ Fed reporter ('Fed whisperer'). First to signal Fed policy shifts. Treat as near-official.",
    ),
    "jukan05": AccountProfile(
        "sector_tech",
        "Semi/tech stock deep dives. INTC, foundry, memory supply chain analysis.",
    ),
    "aleabitoreddit": AccountProfile(
        "general",
        "Broad market commentary, trade ideas, earnings plays.",
    ),
    "zerohedge": AccountProfile(
        "macro",
        "Financial news/commentary. Macro, geopolitical, contrarian. BIAS: bearish/doom-leaning.",
    ),
    "elonmusk": AccountProfile(
        "tech_policy",
        "Tesla/SpaceX/xAI CEO, DOGE. Market-moving when about his companies or policy. Ignore shitposts.",
    ),
    "kobeissiletter": AccountProfile(
        "macro",
        "Macro market analysis newsletter. Charts, data visualization, trade setups.",
    ),
}


def get_profile(username: str) -> AccountProfile | None:
    """Look up an account profile by username (case-insensitive)."""
    return ACCOUNT_PROFILES.get(username.lower())
