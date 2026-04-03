"""Impact scoring for news messages.

Hybrid system: fast-track rules for known-important patterns (econ data releases,
BREAKING from wire sources, large M&A) + additive scoring for everything else.

Backtested 2026-04 against 918k messages from marketfeed.jsonl:
  critical: 0.47% | high: 0.92% | normal: 27% | low: 72% | Stage 2: 1.39%
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from synesis.core.constants import NEWS_SOURCE_RE as _NEWS_SOURCE_RE
from synesis.processing.news.models import UrgencyLevel

# =============================================================================
# Constants
# =============================================================================

# Source account → (class label, score weight)
SOURCE_RELIABILITY: dict[str, tuple[str, int]] = {
    "FirstSquawk": ("wire_relay", 15),
    "DeItaone": ("wire_relay", 15),
    "diegobloomberg": ("wire_relay", 12),
    "LiveSquawk": ("wire_relay", 12),
    "financialjuice": ("data_relay", 10),
    "Newsquawk": ("wire_relay", 10),
    "tradfi": ("curated", 8),
    "unusual_whales": ("curated", 5),
    "tier10k": ("curated", 5),
    "IGSquawk": ("curated", 5),
    "WatcherGuru": ("sensational", 0),
    "cryptounfolded": ("sensational", 0),
    "Zfied": ("unknown", 0),
    # Telegram channels (source_account when no x.com URL in text)
    "disclosetv": ("newswire", 10),
    "unfolded": ("newswire", 10),
    "ZoomerfiedNews": ("newswire", 10),
    "TreeNewsFeed": ("newswire", 10),
    "dbnewsdelayed": ("newswire", 10),
    "thekobeissiletter": ("commentary", 0),
}

# Sources trusted enough for fast-track BREAKING rules
WIRE_RELAY_SOURCES = frozenset(
    {
        "FirstSquawk",
        "DeItaone",
        "diegobloomberg",
        "LiveSquawk",
        "Newsquawk",
        "financialjuice",
        "tradfi",
    }
)

# Score → UrgencyLevel thresholds
CRITICAL_THRESHOLD = 55
HIGH_THRESHOLD = 32
NORMAL_THRESHOLD = 15

# Component caps
_TEXT_PATTERN_CAP = 35
_CONTENT_TYPE_CAP = 20
_MAGNITUDE_CAP = 20

_DOLLAR_MULTIPLIERS: dict[str, float] = {
    "TRILLION": 1e12,
    "BILLION": 1e9,
    "BLN": 1e9,
    "B": 1e9,
    "MILLION": 1e6,
    "MLN": 1e6,
    "M": 1e6,
    "THOUSAND": 1e3,
    "K": 1e3,
}

_LEVEL_RANK: dict[str, int] = {"low": 0, "normal": 1, "high": 2, "critical": 3}

# =============================================================================
# Shared regex (used by both fast-track and scoring)
# =============================================================================

# "*BREAKING: FED CUTS RATES", "JUST IN: OPENAI RAISES $122B", "⚠ BREAKING: IRAN..."
# "🔴US CPI ACTUAL: 2.5%", "🚨 BREAKING: ..."
_BREAKING_RE = re.compile(
    r"\*+\s*BREAKING|^BREAKING[:\s]|⚠\s*BREAKING|JUST\s+IN[:\s-]"
    r"|(?:FLASH|ALERT|URGENT)[:\s]|[🔴🚨]",
    re.IGNORECASE,
)

# Strict econ release format — distinguishes actual data releases from commentary
# YES: "🔴CANADIAN TRADE BALANCE ACTUAL -5.74B (FORECAST -2.5B, PREVIOUS -3.65B)"
# YES: "CHINA (MAR) CPI YOY ACTUAL: 0.3% VS 0.1% PREVIOUS;EST 0.5%"
# YES: "*TAIWAN PRELIM 4Q GDP RISES 12.65% Y/Y; EST. +12.70%"
# NO:  "FED'S WILLIAMS: WAR COULD INCREASE INFLATION" (commentary, no ACTUAL/EST)
_RELEASE_FORMAT_RE = re.compile(
    r"ACTUAL\s*:?\s*[\d.-]"
    r"|(?:EST|FORECAST|FCAST)\s*:?\s*[\d.-]"
    r"|\((?:Y/Y|M/M|Q/Q|MOM|YOY|QOQ)\)"
    r"|\bVS\s+[\d.]+[%KBM]?\s+(?:EST|PREV|PREVIOUS)",
    re.IGNORECASE,
)

# Tier 1: Top macro indicators — move broad markets
# "US CPI (YOY) (MAR) ACTUAL: 2.5%", "US ADP NONFARM EMPLOYMENT CHANGE (MAR) ACTUAL: 62K"
_ECON_T1_RE = re.compile(
    r"\b(?:CPI|PCE|NFP|NON-?FARM\s+(?:PAYROLLS?|EMPLOYMENT)|GDP)\b",
    re.IGNORECASE,
)

# Tier 2: Important but narrower indicators
# "US INITIAL JOBLESS CLAIMS ACTUAL: 202K", "US ISM MANUFACTURING PMI (MAR) ACTUAL: 52.7"
_ECON_T2_RE = re.compile(
    r"\b(?:PPI|RETAIL\s+SALES|INITIAL\s+JOBLESS|JOBLESS\s+CLAIMS|UNEMPLOYMENT\s+RATE"
    r"|PMI|ISM|DURABLE\s+GOODS|HOUSING\s+STARTS|CONSUMER\s+(?:CONFIDENCE|SENTIMENT)"
    r"|INDUSTRIAL\s+PRODUCTION|TRADE\s+BALANCE)\b",
    re.IGNORECASE,
)

# Tier 3: Minor/regional indicators
# "FACTORY ORDERS", "BUILDING PERMITS", "PERSONAL SPENDING"
_ECON_T3_RE = re.compile(
    r"\b(?:FACTORY\s+ORDERS|BUILDING\s+PERMITS|EXISTING\s+HOME|NEW\s+HOME"
    r"|PENDING\s+HOME|WHOLESALE\s+INVENTOR|PERSONAL\s+(?:INCOME|SPENDING))\b",
    re.IGNORECASE,
)

# M&A / corporate action keywords (shared by scoring + fast-track)
# "*NVIDIA INVESTS $2B IN MARVELL", "$5 BLN ANCHOR STAKE", "OPENAI RAISES $122B"
_MNA_RE = re.compile(
    r"\b(?:ACQUIR[EIS]?[DS]?|MERGER|TAKEOVER|BUYOUT|M&A)\b"
    r"|\bSTAKE\b.*[$]|[$].*\bSTAKE\b"
    r"|\b(?:INVESTS?|INVESTMENT|RAISES?)\b.*[$]|[$].*\b(?:INVESTS?|INVESTMENT)\b",
    re.IGNORECASE,
)

# Dollar amounts with explicit magnitude suffix (avoids matching "$141.37/BBL")
# YES: "$5 BLN", "$777 BILLION", "$110B"
# NO:  "$141.37/BBL" (BBL is not a valid suffix)
_DOLLAR_RE = re.compile(
    r"[$]\s*([\d,]+(?:\.\d+)?)\s*(TRILLION|BILLION|BLN|MILLION|MLN|THOUSAND|B|M|K)\b",
    re.IGNORECASE,
)

# =============================================================================
# A. Text Pattern Signals
# =============================================================================

# "*OPENAI ACQUIRED ONLINE TECH NEWS TALK SHOW :WSJ"
_WIRE_PREFIX_RE = re.compile(r"^\*")

# "BREAKING: TRUMP REMOVES BONDI AS ATTORNEY GENERAL, PER CNN"
_BREAKING_ONLY_RE = re.compile(r"\*+\s*BREAKING|^BREAKING[:\s]|⚠\s*BREAKING", re.IGNORECASE)

# "JUST IN: OPENAI RAISES $122,000,000,000 AT $852 BILLION VALUATION"
_JUST_IN_RE = re.compile(r"JUST\s+IN[:\s-]", re.IGNORECASE)

# "OIL ALERT: COULD BRENT CRUDE HIT $135?"
_FLASH_ALERT_RE = re.compile(r"(?:FLASH|ALERT|URGENT)[:\s]", re.IGNORECASE)

# "*EXCLUSIVE - TRUMP APPROVAL FALLS TO 36%, LOWEST SINCE RETURN TO WHITE HOUSE"
_EXCLUSIVE_RE = re.compile(r"(?:EXCLUSIVE|SCOOP)[:\s]", re.IGNORECASE)

# "BRENT OIL PRICE SOARS TO $141.37/BBL, HIGHEST SINCE 2008"
_SUPERLATIVE_RE = re.compile(
    r"\b(?:HIGHEST|WORST|LOWEST|BIGGEST|LARGEST|SMALLEST|RECORD|FIRST\s+TIME)"
    r"\s+(?:SINCE|IN\s+\d+|EVER)\b",
    re.IGNORECASE,
)

# financialjuice data release marker: "🔴US INITIAL JOBLESS CLAIMS ACTUAL 202K"
_RED_CIRCLE_RE = re.compile(r"🔴")

# "SURPRISE FED CUT", "UNEXPECTED RISE IN UNEMPLOYMENT"
_SURPRISE_RE = re.compile(r"\b(?:SURPRISE|UNEXPECTED|SHOCKED)\b", re.IGNORECASE)

# "OIL PRICES SURGED NEARLY 10%", "OVER $777 BILLION WIPED OUT"
_EXTREME_MOVE_RE = re.compile(
    r"\b(?:WIPED\s+OUT|CRASH(?:ES|ED)?|PLUNGE[DS]?|SURGE[DS]?|SOAR(?:S|ED)?"
    r"|SKYROCKET|TUMBLE[DS]?|SPIKE[DS]?|COLLAPSE[DS]?)\b",
    re.IGNORECASE,
)


_TEXT_SIGNALS: list[tuple[re.Pattern[str], int, str]] = [
    (_WIRE_PREFIX_RE, 15, "wire_prefix"),
    (_BREAKING_ONLY_RE, 12, "breaking"),
    (_JUST_IN_RE, 12, "just_in"),
    (_FLASH_ALERT_RE, 12, "flash_alert"),
    (_EXCLUSIVE_RE, 10, "exclusive"),
    (_SUPERLATIVE_RE, 10, "superlative"),
    (_RED_CIRCLE_RE, 12, "red_circle"),
    (_SURPRISE_RE, 10, "surprise"),
    (_EXTREME_MOVE_RE, 8, "extreme_move"),
]


def _text_pattern_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    for pattern, points, label in _TEXT_SIGNALS:
        if pattern.search(text):
            score += points
            reasons.append(f"{label}:+{points}")
    return min(score, _TEXT_PATTERN_CAP), reasons


# =============================================================================
# B. Source Reliability
# =============================================================================


def _resolve_source(source_account: str, text: str) -> str:
    """Resolve the actual news source for reliability scoring.

    In production, source_account is the Telegram channel name (e.g. "marketfeed"),
    not the wire source. The wire source (DeItaone, FirstSquawk) is embedded in
    the x.com URL within the text. Try source_account first, fall back to URL extraction.
    """
    if source_account in SOURCE_RELIABILITY:
        return source_account
    m = _NEWS_SOURCE_RE.search(text)
    if m and m.group(1) in SOURCE_RELIABILITY:
        return m.group(1)
    return source_account


def _source_score(source: str) -> tuple[int, str]:
    if source in SOURCE_RELIABILITY:
        cls, weight = SOURCE_RELIABILITY[source]
        return weight, cls
    return 0, "unknown"


# =============================================================================
# C. Content-Type Signals
# =============================================================================

# "RATE CUT", "RATE HIKE", "RATE DECISION"
_RATE_DECISION_RE = re.compile(
    r"\bRATE\s+(?:CUT|HIKE|DECISION|UNCHANGED)\b",
    re.IGNORECASE,
)

# "RUSSIA WILL ASK THE US AND ISRAEL TO ENSURE A CEASEFIRE...IRAN'S BUSHEHR NUCLEAR SITE"
_GEOPOLITICAL_RE = re.compile(
    r"\b(?:AIR\s*STRIKE[S]?|INVAD(?:E[DS]?|ING)|INVASION|SANCTION[S]?"
    r"|DECLARES?\s+WAR|MISSILE[S]?|NUCLEAR|BOMB(?:S|ED|ING)?"
    r"|MILITARY\s+(?:ACTION|OPERATION))\b",
    re.IGNORECASE,
)

# "SPACEX HAS OFFICIALLY FILED FOR AN IPO IN JUNE"
_IPO_RE = re.compile(
    r"\b(?:IPO|GOES?\s+PUBLIC|(?:FILES?|FILED)\s+(?:FOR\s+)?IPO)\b",
    re.IGNORECASE,
)

# "EARNINGS BEAT", "$NKE NIKE Q3 EPS $0.35, EST. $0.31"
_EARNINGS_RE = re.compile(
    r"\b(?:EARNINGS|EPS|REVENUE)\b.*\b(?:BEAT|MISS|SURPASS|EXCEED|BELOW)\b",
    re.IGNORECASE,
)

# "TRUMP IMPOSES 25% TARIFFS ON CHINA", "TRADE WAR"
_TARIFF_RE = re.compile(
    r"\b(?:TARIFF[S]?|TRADE\s+(?:WAR|DEAL|BAN)|IMPORT\s+DUT(?:Y|IES))\b",
    re.IGNORECASE,
)

# "RED LOBSTER TO BRING BACK ENDLESS SHRIMP THAT BROUGHT IT TO BANKRUPTCY"
_BANKRUPTCY_RE = re.compile(
    r"\b(?:BANKRUPT(?:CY)?|DEFAULT[S]?|DOWNGRADE[DS]?|INSOLVEN(?:T|CY)?)\b",
    re.IGNORECASE,
)

# "ELI LILLY'S FOUNDAYO IS NOW FDA APPROVED FOR ADULTS WITH OBESITY"
_FDA_RE = re.compile(r"\bFDA\s+(?:APPROV|REJECT|BLOCK|HALT|BAN)\b", re.IGNORECASE)

# "ORACLE CUTS 18% OF ITS WORKFORCE, STOCK RISES 6% DUE TO AI-RELATED LAYOFFS"
_LAYOFFS_RE = re.compile(r"\b(?:LAYOFF[S]?|CUT[S]?\s+\d+.*JOBS|RESTRUCTUR)\b", re.IGNORECASE)


_CONTENT_SIGNALS: list[tuple[re.Pattern[str], int, str]] = [
    (_MNA_RE, 15, "mna"),
    (_RATE_DECISION_RE, 15, "rate_decision"),
    (_GEOPOLITICAL_RE, 15, "geopolitical"),
    (_IPO_RE, 10, "ipo"),
    (_EARNINGS_RE, 10, "earnings_beat_miss"),
    (_TARIFF_RE, 10, "tariff"),
    (_BANKRUPTCY_RE, 10, "bankruptcy"),
    (_FDA_RE, 10, "fda"),
    (_LAYOFFS_RE, 6, "layoffs"),
]

# Tiered econ data: (pattern, base_points, label_prefix) — checked in priority order
_ECON_TIERS: list[tuple[re.Pattern[str], int, str]] = [
    (_ECON_T1_RE, 15, "econ_t1"),
    (_ECON_T2_RE, 10, "econ_t2"),
    (_ECON_T3_RE, 6, "econ_t3"),
]


def _content_type_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    for pattern, points, label in _CONTENT_SIGNALS:
        if pattern.search(text):
            score += points
            reasons.append(f"{label}:+{points}")

    # Tiered economic data (+5 bonus if release format present)
    has_release = bool(_RELEASE_FORMAT_RE.search(text))
    bonus = 5 if has_release else 0
    for pattern, base, label in _ECON_TIERS:
        if pattern.search(text):
            pts = base + bonus
            tag = f"{label}_release" if has_release else label
            score += pts
            reasons.append(f"{tag}:+{pts}")
            break  # Only match highest tier

    return min(score, _CONTENT_TYPE_CAP), reasons


# =============================================================================
# D. Magnitude Signals
# =============================================================================

# "OIL PRICES SURGED NEARLY 10%", "BLUE OWL SHARES FALL 7.6%"
_PCT_MOVE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%.*?\b(?:DROP|SURGE|CRASH|WIPE|GAIN|LOSS|RALLY|FALL|RISE|PLUNGE|TUMBLE|SPIKE|SOAR)\b"
    r"|\b(?:DROP|SURGE|CRASH|WIPE|GAIN|LOSS|RALLY|FALL|RISE|PLUNGE|TUMBLE|SPIKE|SOAR)\w*\s+.*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _parse_dollar(match: re.Match[str]) -> float:
    raw_str = match.group(1)
    if not raw_str:
        return 0.0
    raw = float(raw_str.replace(",", ""))
    suffix = (match.group(2) or "").upper()
    return raw * _DOLLAR_MULTIPLIERS.get(suffix, 1.0)


def _max_dollar_amount(text: str) -> float:
    """Extract the largest dollar amount with magnitude suffix from text."""
    result = 0.0
    for m in _DOLLAR_RE.finditer(text):
        try:
            result = max(result, _parse_dollar(m))
        except (ValueError, AttributeError):
            continue
    return result


def _magnitude_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    max_dollar = _max_dollar_amount(text)

    # Dollar thresholds: (min_amount, points, format_fn)
    dollar_tiers: list[tuple[float, int, Callable[[float], str]]] = [
        (100e9, 20, lambda d: f"${d / 1e9:.0f}B"),
        (10e9, 16, lambda d: f"${d / 1e9:.0f}B"),
        (1e9, 12, lambda d: f"${d / 1e9:.1f}B"),
        (100e6, 8, lambda d: f"${d / 1e6:.0f}M"),
        (10e6, 4, lambda d: f"${d / 1e6:.0f}M"),
    ]
    for threshold, points, fmt in dollar_tiers:
        if max_dollar >= threshold:
            score += points
            reasons.append(f"dollar:{fmt(max_dollar)}:+{points}")
            break

    pct_match = _PCT_MOVE_RE.search(text)
    if pct_match:
        pct_str = pct_match.group(1) or pct_match.group(2)
        if pct_str:
            pct = float(pct_str)
            if pct >= 5:
                score += 10
                reasons.append(f"pct_move:{pct}%:+10")
            elif pct >= 2:
                score += 6
                reasons.append(f"pct_move:{pct}%:+6")

    return min(score, _MAGNITUDE_CAP), reasons


# =============================================================================
# E. Suppressors (uncapped negative)
# =============================================================================

_RT_RE = re.compile(r"^RT\s+@", re.IGNORECASE)

_PROMO_RE = re.compile(
    r"\b(?:GIVEAWAY|AIRDROP|SUBSCRIBE|FOLLOW\s+(?:US|ME|FOR)|SIGN\s*UP|REFERRAL"
    r"|PROMO|LIMITED\s+TIME|ACT\s+NOW|DON'?T\s+MISS|LAST\s+CHANCE)\b"
    r"|(?:bit\.ly|tinyurl\.com)/",
    re.IGNORECASE,
)

_OPINION_RE = re.compile(
    r"\b(?:I\s+THINK|IN\s+MY\s+(?:VIEW|OPINION)|MY\s+TAKE|IMO|IMHO|COMMENTARY)\b",
    re.IGNORECASE,
)

# Social reply without financial terms: "@user nice!" but NOT "@user WHAT ABOUT CPI?"
_SOCIAL_REPLY_RE = re.compile(
    r"^@\w+\s+(?!.*\b(?:STOCK|MARKET|PRICE|TRADE|EARNING|GDP|CPI|RATE|FED"
    r"|INFLATION|TARIFF|BILLION|MILLION|PERCENT|%)\b)",
    re.IGNORECASE,
)

# "Follow us on X for even faster headlines!"
_FOLLOW_PROMO_RE = re.compile(r"(?:Follow\s+us|faster\s+headlines)", re.IGNORECASE)

# "🌅 Market News Digest", "8-HOUR NEWS DIGEST", "MORNING ALL!", "━━━"
_DIGEST_RE = re.compile(
    r"Market\s+News\s+Digest|NEWS\s+DIGEST|8-HOUR|DAILY\s+DIGEST"
    r"|MORNING\s+(?:ALL|BRIEF)|RECAP|WEEKLY\s+WRAP"
    r"|^📰\s*.*DIGEST|^━━━",
    re.IGNORECASE | re.MULTILINE,
)

# "YESTERDAY'S RECAP", "LAST WEEK PERFORMANCE"
_STALE_RE = re.compile(r"\b(?:YESTERDAY|LAST\s+WEEK|LAST\s+MONTH)\b", re.IGNORECASE)


_SUPPRESSOR_SIGNALS: list[tuple[re.Pattern[str], int, str]] = [
    (_RT_RE, -15, "rt"),
    (_PROMO_RE, -20, "promo"),
    (_OPINION_RE, -8, "opinion"),
    (_SOCIAL_REPLY_RE, -10, "social_reply"),
    (_FOLLOW_PROMO_RE, -20, "follow_promo"),
    (_DIGEST_RE, -15, "digest"),
    (_STALE_RE, -5, "stale"),
]


def _suppressor_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    for pattern, penalty, label in _SUPPRESSOR_SIGNALS:
        if pattern.search(text):
            score += penalty
            reasons.append(f"{label}:{penalty}")
    return score, reasons


# =============================================================================
# Fast-Track Rules
# =============================================================================


def _fast_track(text: str, source: str) -> tuple[UrgencyLevel, str] | None:
    """Check fast-track rules that bypass scoring.

    Returns (level, reason) or None. Only upgrades, never downgrades.
    Econ fast-tracks require a known source to avoid promoting Telegram digests.
    """
    is_known = source in SOURCE_RELIABILITY
    has_release = bool(_RELEASE_FORMAT_RE.search(text))

    # Tier 1 econ release from known source → CRITICAL
    if is_known and has_release and _ECON_T1_RE.search(text):
        return (UrgencyLevel.critical, "fast_track:econ_t1_release")

    # Tier 2 econ release from known source → HIGH
    if is_known and has_release and _ECON_T2_RE.search(text):
        return (UrgencyLevel.high, "fast_track:econ_t2_release")

    # M&A/corporate action with $1B+ → CRITICAL (stock-specific, highly actionable)
    amt = _max_dollar_amount(text)
    if _MNA_RE.search(text) and amt >= 1e9:
        return (UrgencyLevel.critical, f"fast_track:mna_${amt / 1e9:.0f}B")

    # BREAKING/JUST IN from known source → HIGH
    if is_known and _BREAKING_RE.search(text):
        return (UrgencyLevel.high, f"fast_track:breaking:{source}")

    return None


# =============================================================================
# Public API
# =============================================================================


@dataclass
class ImpactResult:
    """Result of impact scoring for a news message."""

    score: int  # 0-100 additive score
    urgency: UrgencyLevel  # Derived from score + fast-track
    components: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def compute_impact_score(text: str, source_account: str) -> ImpactResult:
    """Compute impact score and urgency level for a news message.

    1. Resolve actual news source (channel name or extracted from x.com URL)
    2. Fast-track rules checked first (can only upgrade level)
    3. Additive scoring across 5 signal components
    4. Fast-track override applied if it results in a higher level
    """
    # Resolve source: "marketfeed" channel → extract "DeItaone" from x.com URL
    source = _resolve_source(source_account, text)

    ft = _fast_track(text, source)

    a_score, a_reasons = _text_pattern_score(text)
    b_score, b_class = _source_score(source)
    c_score, c_reasons = _content_type_score(text)
    d_score, d_reasons = _magnitude_score(text)
    e_score, e_reasons = _suppressor_score(text)

    raw = a_score + b_score + c_score + d_score + e_score
    score = max(0, min(100, raw))

    if score >= CRITICAL_THRESHOLD:
        urgency = UrgencyLevel.critical
    elif score >= HIGH_THRESHOLD:
        urgency = UrgencyLevel.high
    elif score >= NORMAL_THRESHOLD:
        urgency = UrgencyLevel.normal
    else:
        urgency = UrgencyLevel.low

    reasons = (
        [f"source:{source}:{b_class}:+{b_score}"] + a_reasons + c_reasons + d_reasons + e_reasons
    )

    # Fast-track: only upgrade
    if ft is not None:
        ft_urgency, ft_reason = ft
        if _LEVEL_RANK[ft_urgency.value] > _LEVEL_RANK[urgency.value]:
            urgency = ft_urgency
            reasons.insert(0, ft_reason)

    return ImpactResult(
        score=score,
        urgency=urgency,
        components={
            "text_pattern": a_score,
            "source": b_score,
            "content_type": c_score,
            "magnitude": d_score,
            "suppressor": e_score,
        },
        reasons=reasons,
    )
