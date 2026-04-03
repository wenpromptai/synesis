"""Test the proposed impact scoring system against historical marketfeed data.

This is the reference implementation for src/synesis/processing/news/impact_scorer.py.
Run: python3 scripts/test_impact_scorer.py [max_lines]  (0 or omit = all)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# =============================================================================
# Constants
# =============================================================================

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
    "WatcherGuru": ("sensational", -2),
    "cryptounfolded": ("sensational", -5),
    "Zfied": ("unknown", 0),
}

# Sources trusted enough for fast-track rules
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

CRITICAL_THRESHOLD = 55
HIGH_THRESHOLD = 32
NORMAL_THRESHOLD = 15

TEXT_PATTERN_CAP = 35
CONTENT_TYPE_CAP = 20
MAGNITUDE_CAP = 20

DOLLAR_MULTIPLIERS = {
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

# =============================================================================
# Shared regex patterns (used by both fast-track and scoring)
# =============================================================================

# Breaking/urgent markers (shared by fast-track and scoring)
# "*BREAKING: ...", "JUST IN: OPENAI RAISES $122,000,000,000...", "⚠ BREAKING: IRAN..."
_BREAKING_RE = re.compile(
    r"\*+\s*BREAKING|^BREAKING[:\s]|⚠\s*BREAKING|JUST\s+IN[:\s-]|(?:FLASH|ALERT|URGENT)[:\s]",
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
# "US CPI (YOY) (MAR) ACTUAL: 2.5%", "US ADP NONFARM EMPLOYMENT CHANGE (MAR) ACTUAL: 62K", "*US 3Q GDP RISES 4.4%"
_ECON_T1_RE = re.compile(
    r"\b(?:CPI|PCE|NFP|NON-?FARM\s+(?:PAYROLLS?|EMPLOYMENT)|GDP)\b",
    re.IGNORECASE,
)

# Tier 2: Important but narrower indicators
# "US INITIAL JOBLESS CLAIMS ACTUAL: 202K", "US ISM MANUFACTURING PMI (MAR) ACTUAL: 52.7", "US RETAIL SALES MOM ACTUAL 0.6%"
_ECON_T2_RE = re.compile(
    r"\b(?:PPI|RETAIL\s+SALES|INITIAL\s+JOBLESS|JOBLESS\s+CLAIMS|UNEMPLOYMENT\s+RATE"
    r"|PMI|ISM|DURABLE\s+GOODS|HOUSING\s+STARTS|CONSUMER\s+(?:CONFIDENCE|SENTIMENT)"
    r"|INDUSTRIAL\s+PRODUCTION|TRADE\s+BALANCE)\b",
    re.IGNORECASE,
)

# Tier 3: Minor/regional indicators
# e.g. "FACTORY ORDERS", "BUILDING PERMITS", "PERSONAL SPENDING"
_ECON_T3_RE = re.compile(
    r"\b(?:FACTORY\s+ORDERS|BUILDING\s+PERMITS|EXISTING\s+HOME|NEW\s+HOME"
    r"|PENDING\s+HOME|WHOLESALE\s+INVENTOR|PERSONAL\s+(?:INCOME|SPENDING))\b",
    re.IGNORECASE,
)

# =============================================================================
# Fast-Track Rules
# =============================================================================


def _fast_track(text: str, source: str) -> tuple[str, str] | None:
    """Check fast-track rules that bypass scoring.

    Returns (level, reason) or None. Only upgrades, never downgrades.
    Econ fast-tracks require a known source to avoid promoting Telegram digests.
    """
    is_known = source in SOURCE_RELIABILITY
    has_release = bool(_RELEASE_FORMAT_RE.search(text))

    # Tier 1 econ release from known source → CRITICAL
    if is_known and has_release:
        if _ECON_T1_RE.search(text):
            return ("critical", "fast_track:econ_t1_release")
        if _ECON_T2_RE.search(text):
            return ("high", "fast_track:econ_t2_release")

    # M&A/corporate action with $1B+ dollar amount → CRITICAL
    # "INVESTS $2B IN MARVELL", "RAISES $122B", "$5 BLN ANCHOR STAKE"
    if _MNA_RE.search(text):
        max_dollar = 0.0
        for m in _DOLLAR_RE.finditer(text):
            try:
                max_dollar = max(max_dollar, _parse_dollar(m))
            except (ValueError, AttributeError):
                continue
        if max_dollar >= 1e9:
            return ("critical", f"fast_track:mna_${max_dollar / 1e9:.0f}B")

    # BREAKING/JUST IN from wire sources → HIGH
    if source in WIRE_RELAY_SOURCES and _BREAKING_RE.search(text):
        return ("high", f"fast_track:breaking_wire:{source}")

    return None


# =============================================================================
# A. Text Pattern Signals (capped)
# =============================================================================

# Bloomberg terminal relay: "*OPENAI ACQUIRED ONLINE TECH NEWS TALK SHOW :WSJ"
_WIRE_PREFIX_RE = re.compile(r"^\*")

# "BREAKING: TRUMP REMOVES BONDI AS ATTORNEY GENERAL, PER CNN"
# "⚠ BREAKING: IRAN: DRAFTING PROTOCOL WITH OMAN FOR HORMUZ STRAIT TRAFFIC"
_BREAKING_ONLY_RE = re.compile(r"\*+\s*BREAKING|^BREAKING[:\s]|⚠\s*BREAKING", re.IGNORECASE)

# "JUST IN: OPENAI RAISES $122,000,000,000 AT $852 BILLION VALUATION"
_JUST_IN_RE = re.compile(r"JUST\s+IN[:\s-]", re.IGNORECASE)

# "OIL ALERT: COULD BRENT CRUDE HIT $135?"
_FLASH_ALERT_RE = re.compile(r"(?:FLASH|ALERT|URGENT)[:\s]", re.IGNORECASE)

# "*EXCLUSIVE - TRUMP APPROVAL FALLS TO 36%, LOWEST SINCE RETURN TO WHITE HOUSE"
_EXCLUSIVE_RE = re.compile(r"(?:EXCLUSIVE|SCOOP)[:\s]", re.IGNORECASE)

# Historical comparison = outlier signal
# "BRENT OIL PRICE SOARS TO $141.37/BBL, HIGHEST SINCE 2008"
_SUPERLATIVE_RE = re.compile(
    r"\b(?:HIGHEST|WORST|LOWEST|BIGGEST|LARGEST|SMALLEST|RECORD|FIRST\s+TIME)\s+(?:SINCE|IN\s+\d+|EVER)\b",
    re.IGNORECASE,
)

# financialjuice data release marker: "🔴US INITIAL JOBLESS CLAIMS ACTUAL 202K (FORECAST 212K)"
_RED_CIRCLE_RE = re.compile(r"🔴")

# "SURPRISE FED CUT", "UNEXPECTED RISE IN UNEMPLOYMENT"
_SURPRISE_RE = re.compile(r"\b(?:SURPRISE|UNEXPECTED|SHOCKED)\b", re.IGNORECASE)

# Extreme price action: "OIL PRICES SURGED NEARLY 10%", "OVER $777 BILLION WIPED OUT"
_EXTREME_MOVE_RE = re.compile(
    r"\b(?:WIPED\s+OUT|CRASH(?:ES|ED)?|PLUNGE[DS]?|SURGE[DS]?|SOAR[ES]?"
    r"|SKYROCKET|TUMBLE[DS]?|SPIKE[DS]?|COLLAPSE[DS]?)\b",
    re.IGNORECASE,
)


def _text_pattern_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if _WIRE_PREFIX_RE.search(text):
        score += 15
        reasons.append("wire_prefix:+15")
    if _BREAKING_ONLY_RE.search(text):
        score += 12
        reasons.append("breaking:+12")
    if _JUST_IN_RE.search(text):
        score += 10
        reasons.append("just_in:+10")
    if _FLASH_ALERT_RE.search(text):
        score += 12
        reasons.append("flash_alert:+12")
    if _EXCLUSIVE_RE.search(text):
        score += 10
        reasons.append("exclusive:+10")
    if _SUPERLATIVE_RE.search(text):
        score += 10
        reasons.append("superlative:+10")
    if _RED_CIRCLE_RE.search(text):
        score += 6
        reasons.append("red_circle:+6")
    if _SURPRISE_RE.search(text):
        score += 10
        reasons.append("surprise:+10")
    if _EXTREME_MOVE_RE.search(text):
        score += 8
        reasons.append("extreme_move:+8")

    return min(score, TEXT_PATTERN_CAP), reasons


# =============================================================================
# B. Source Reliability
# =============================================================================


def _source_score(source: str) -> tuple[int, str]:
    if source in SOURCE_RELIABILITY:
        cls, weight = SOURCE_RELIABILITY[source]
        return weight, cls
    return 0, "unknown"


# =============================================================================
# C. Content-Type Signals (capped)
# =============================================================================

# "*NVIDIA INVESTS $2B IN MARVELL", "SPACEX...FOR $5 BLN ANCHOR STAKE"
# "OPENAI RAISES $122,000,000,000 AT $852 BILLION VALUATION"
_MNA_RE = re.compile(
    r"\b(?:ACQUIR[EIS]?[DS]?|MERGER|TAKEOVER|BUYOUT|M&A)\b"
    r"|\bSTAKE\b.*[$]|[$].*\bSTAKE\b"
    r"|\b(?:INVESTS?|INVESTMENT|RAISES?)\b.*[$]|[$].*\b(?:INVESTS?|INVESTMENT)\b",
    re.IGNORECASE,
)

# e.g. "RATE CUT", "RATE HIKE", "RATE DECISION"
# Note: FOMC/FED handled by _ECON_T1_RE to avoid double-counting
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

# "SPACEX HAS OFFICIALLY FILED FOR AN IPO IN JUNE", "*INSILICO SHARES RISE 45% IN HONG KONG AFTER $293M IPO"
_IPO_RE = re.compile(r"\b(?:IPO|GOES?\s+PUBLIC|(?:FILES?|FILED)\s+(?:FOR\s+)?IPO)\b", re.IGNORECASE)

# "EARNINGS BEAT", "$NKE NIKE Q3 EPS $0.35, EST. $0.31"
_EARNINGS_RE = re.compile(
    r"\b(?:EARNINGS|EPS|REVENUE)\b.*\b(?:BEAT|MISS|SURPASS|EXCEED|BELOW)\b", re.IGNORECASE
)

# "TRUMP IMPOSES 25% TARIFFS ON CHINA", "TRADE WAR", "IMPORT DUTY"
_TARIFF_RE = re.compile(
    r"\b(?:TARIFF[S]?|TRADE\s+(?:WAR|DEAL|BAN)|IMPORT\s+DUT(?:Y|IES))\b", re.IGNORECASE
)

# "RED LOBSTER TO BRING BACK ENDLESS SHRIMP THAT BROUGHT IT TO BANKRUPTCY"
_BANKRUPTCY_RE = re.compile(
    r"\b(?:BANKRUPT(?:CY)?|DEFAULT[S]?|DOWNGRADE[DS]?|INSOLVEN)\b", re.IGNORECASE
)

# "ELI LILLY'S FOUNDAYO IS NOW FDA APPROVED FOR ADULTS WITH OBESITY"
_FDA_RE = re.compile(r"\bFDA\s+(?:APPROV|REJECT|BLOCK|HALT|BAN)\b", re.IGNORECASE)

# "ORACLE CUTS 18% OF ITS WORKFORCE, STOCK RISES 6% DUE TO AI-RELATED LAYOFFS"
_LAYOFFS_RE = re.compile(r"\b(?:LAYOFF[S]?|CUT[S]?\s+\d+.*JOBS|RESTRUCTUR)\b", re.IGNORECASE)


def _content_type_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    # Event-type patterns
    if _MNA_RE.search(text):
        score += 15
        reasons.append("mna:+15")
    if _RATE_DECISION_RE.search(text):
        score += 15
        reasons.append("rate_decision:+15")
    if _GEOPOLITICAL_RE.search(text):
        score += 15
        reasons.append("geopolitical:+15")
    if _IPO_RE.search(text):
        score += 10
        reasons.append("ipo:+10")
    if _EARNINGS_RE.search(text):
        score += 10
        reasons.append("earnings_beat_miss:+10")
    if _TARIFF_RE.search(text):
        score += 10
        reasons.append("tariff:+10")
    if _BANKRUPTCY_RE.search(text):
        score += 10
        reasons.append("bankruptcy:+10")
    if _FDA_RE.search(text):
        score += 10
        reasons.append("fda:+10")
    if _LAYOFFS_RE.search(text):
        score += 6
        reasons.append("layoffs:+6")

    # Tiered economic data (uses shared _ECON_T*_RE patterns)
    has_release = bool(_RELEASE_FORMAT_RE.search(text))
    release_bonus = 5 if has_release else 0

    if _ECON_T1_RE.search(text):
        pts = 15 + release_bonus
        tag = "econ_t1_release" if has_release else "econ_t1"
        score += pts
        reasons.append(f"{tag}:+{pts}")
    elif _ECON_T2_RE.search(text):
        pts = 10 + release_bonus
        tag = "econ_t2_release" if has_release else "econ_t2"
        score += pts
        reasons.append(f"{tag}:+{pts}")
    elif _ECON_T3_RE.search(text):
        pts = 6 + release_bonus
        tag = "econ_t3_release" if has_release else "econ_t3"
        score += pts
        reasons.append(f"{tag}:+{pts}")

    return min(score, CONTENT_TYPE_CAP), reasons


# =============================================================================
# D. Magnitude Signals (capped)
# =============================================================================

# Dollar amounts with explicit magnitude suffix — requires B/BLN/BILLION/M/etc.
# YES: "$5 BLN ANCHOR STAKE", "$777 BILLION WIPED OUT", "$110B IN NEW INVESTMENT"
# NO:  "$141.37/BBL" (no valid suffix — BBL is a commodity unit, not matched)
_DOLLAR_RE = re.compile(
    r"[$]\s*([\d,]+(?:\.\d+)?)\s*(TRILLION|BILLION|BLN|MILLION|MLN|THOUSAND|B|M|K)\b",
    re.IGNORECASE,
)

# Percentage moves near action verbs (bidirectional: "10% SURGE" or "SURGES 10%")
# "OIL PRICES SURGED NEARLY 10%", "BLUE OWL SHARES FALL 7.6%", "$777B WIPED...5% LOSS"
_PCT_MOVE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%.*?\b(?:DROP|SURGE|CRASH|WIPE|GAIN|LOSS|RALLY|FALL|RISE|PLUNGE|TUMBLE|SPIKE|SOAR)\b"
    r"|\b(?:DROP|SURGE|CRASH|WIPE|GAIN|LOSS|RALLY|FALL|RISE|PLUNGE|TUMBLE|SPIKE|SOAR)\w*\s+.*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def _parse_dollar(match: re.Match) -> float:
    raw_str = match.group(1)
    if not raw_str:
        return 0.0
    raw = float(raw_str.replace(",", ""))
    suffix = (match.group(2) or "").upper()
    return raw * DOLLAR_MULTIPLIERS.get(suffix, 1.0)


def _magnitude_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    # Largest dollar amount
    max_dollar = 0.0
    for m in _DOLLAR_RE.finditer(text):
        try:
            val = _parse_dollar(m)
        except (ValueError, AttributeError):
            continue
        max_dollar = max(max_dollar, val)

    if max_dollar >= 100e9:
        score += 20
        reasons.append(f"dollar:${max_dollar / 1e9:.0f}B:+20")
    elif max_dollar >= 10e9:
        score += 16
        reasons.append(f"dollar:${max_dollar / 1e9:.0f}B:+16")
    elif max_dollar >= 1e9:
        score += 12
        reasons.append(f"dollar:${max_dollar / 1e9:.1f}B:+12")
    elif max_dollar >= 100e6:
        score += 8
        reasons.append(f"dollar:${max_dollar / 1e6:.0f}M:+8")
    elif max_dollar >= 10e6:
        score += 4
        reasons.append(f"dollar:${max_dollar / 1e6:.0f}M:+4")

    # Percentage moves near action verbs
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

    return min(score, MAGNITUDE_CAP), reasons


# =============================================================================
# E. Suppressors (uncapped negative)
# =============================================================================

# e.g. "RT @someuser: $SPY will moon"
_RT_RE = re.compile(r"^RT\s+@", re.IGNORECASE)

# Spam/promo: "GIVEAWAY", "SUBSCRIBE", "bit.ly/xyz"
_PROMO_RE = re.compile(
    r"\b(?:GIVEAWAY|AIRDROP|SUBSCRIBE|FOLLOW\s+(?:US|ME|FOR)|SIGN\s*UP|REFERRAL"
    r"|PROMO|LIMITED\s+TIME|ACT\s+NOW|DON'?T\s+MISS|LAST\s+CHANCE)\b"
    r"|(?:bit\.ly|tinyurl\.com)/",
    re.IGNORECASE,
)

# e.g. "I THINK $SPY WILL...", "IN MY OPINION...", "COMMENTARY"
_OPINION_RE = re.compile(
    r"\b(?:I\s+THINK|IN\s+MY\s+(?:VIEW|OPINION)|MY\s+TAKE|IMO|IMHO|COMMENTARY)\b",
    re.IGNORECASE,
)

# Social reply without financial terms: "@user nice pic!" but NOT "@user WHAT ABOUT CPI?"
_SOCIAL_REPLY_RE = re.compile(
    r"^@\w+\s+(?!.*\b(?:STOCK|MARKET|PRICE|TRADE|EARNING|GDP|CPI|RATE|FED"
    r"|INFLATION|TARIFF|BILLION|MILLION|PERCENT|%)\b)",
    re.IGNORECASE,
)

# e.g. "Follow us on X for even faster headlines!"
_FOLLOW_PROMO_RE = re.compile(r"(?:Follow\s+us|faster\s+headlines)", re.IGNORECASE)

# e.g. "🌅 Market News Digest", "8-HOUR NEWS DIGEST", "MORNING ALL!", "━━━"
_DIGEST_RE = re.compile(
    r"Market\s+News\s+Digest|NEWS\s+DIGEST|8-HOUR|DAILY\s+DIGEST"
    r"|MORNING\s+(?:ALL|BRIEF)|RECAP|WEEKLY\s+WRAP"
    r"|^📰\s*.*DIGEST|^━━━",
    re.IGNORECASE | re.MULTILINE,
)

# e.g. "YESTERDAY'S RECAP", "LAST WEEK PERFORMANCE"
_STALE_RE = re.compile(r"\b(?:YESTERDAY|LAST\s+WEEK|LAST\s+MONTH)\b", re.IGNORECASE)


def _suppressor_score(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    if _RT_RE.search(text):
        score -= 15
        reasons.append("rt:-15")
    if _PROMO_RE.search(text):
        score -= 20
        reasons.append("promo:-20")
    if _OPINION_RE.search(text):
        score -= 8
        reasons.append("opinion:-8")
    if _SOCIAL_REPLY_RE.search(text):
        score -= 10
        reasons.append("social_reply:-10")
    if _FOLLOW_PROMO_RE.search(text):
        score -= 20
        reasons.append("follow_promo:-20")
    if _DIGEST_RE.search(text):
        score -= 15
        reasons.append("digest:-15")
    if _STALE_RE.search(text):
        score -= 5
        reasons.append("stale:-5")

    return score, reasons


# =============================================================================
# Main Scorer
# =============================================================================

_LEVEL_RANK = {"low": 0, "normal": 1, "high": 2, "critical": 3}


@dataclass
class ImpactResult:
    score: int
    level: str  # critical/high/normal/low
    components: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def compute_impact_score(text: str, source: str) -> ImpactResult:
    """Compute impact score and level for a news message.

    1. Fast-track rules checked first (can only upgrade level)
    2. Additive scoring across 5 signal components
    3. Fast-track override applied if it results in a higher level
    """
    ft = _fast_track(text, source)

    a_score, a_reasons = _text_pattern_score(text)
    b_score, b_class = _source_score(source)
    c_score, c_reasons = _content_type_score(text)
    d_score, d_reasons = _magnitude_score(text)
    e_score, e_reasons = _suppressor_score(text)

    raw = a_score + b_score + c_score + d_score + e_score
    score = max(0, min(100, raw))

    if score >= CRITICAL_THRESHOLD:
        level = "critical"
    elif score >= HIGH_THRESHOLD:
        level = "high"
    elif score >= NORMAL_THRESHOLD:
        level = "normal"
    else:
        level = "low"

    reasons = (
        [f"source:{source}:{b_class}:+{b_score}"] + a_reasons + c_reasons + d_reasons + e_reasons
    )

    # Fast-track: only upgrade
    if ft is not None:
        ft_level, ft_reason = ft
        if _LEVEL_RANK[ft_level] > _LEVEL_RANK[level]:
            level = ft_level
            reasons.insert(0, ft_reason)

    return ImpactResult(
        score=score,
        level=level,
        components={
            "text_pattern": a_score,
            "source": b_score,
            "content_type": c_score,
            "magnitude": d_score,
            "suppressor": e_score,
        },
        reasons=reasons,
    )


# =============================================================================
# CLI Analysis (test harness only)
# =============================================================================


def extract_source(text: str) -> str:
    """Extract Twitter source from x.com URL in message text."""
    m = re.search(r"x\.com/(\w+)/status", text)
    return m.group(1) if m else "unknown"


def main() -> None:
    filepath = "data/marketfeed.jsonl"
    max_lines = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    total = 0
    level_counts: Counter[str] = Counter()
    source_level: dict[str, Counter[str]] = defaultdict(Counter)
    signal_hits: Counter[str] = Counter()
    score_histogram: Counter[int] = Counter()

    critical_examples: list[tuple[int, str, str, list[str]]] = []
    high_examples: list[tuple[int, str, str, list[str]]] = []
    borderline_miss: list[tuple[int, str, str, list[str]]] = []

    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            obj = json.loads(line)
            text = obj["text"]
            source = extract_source(text)

            result = compute_impact_score(text, source)
            total += 1
            level_counts[result.level] += 1
            source_level[source][result.level] += 1

            for r in result.reasons:
                signal_hits[r.split(":")[0]] += 1

            score_histogram[(result.score // 5) * 5] += 1

            short = text[:120]
            if result.level == "critical" and len(critical_examples) < 20:
                critical_examples.append((result.score, source, short, result.reasons))
            elif result.level == "high" and len(high_examples) < 20:
                high_examples.append((result.score, source, short, result.reasons))
            elif 25 <= result.score <= 31 and len(borderline_miss) < 15:
                borderline_miss.append((result.score, source, short, result.reasons))

    # --- Output ---
    print(f"\n{'=' * 80}")
    print(f"IMPACT SCORING ANALYSIS — {total:,} messages")
    print(f"{'=' * 80}")

    print("\n## Level Distribution")
    for lvl in ("critical", "high", "normal", "low"):
        cnt = level_counts[lvl]
        print(f"  {lvl:10s}: {cnt:>8,} ({cnt / total * 100:5.2f}%)")
    s2 = level_counts["critical"] + level_counts["high"]
    print(f"  {'→ Stage 2':10s}: {s2:>8,} ({s2 / total * 100:5.2f}%) pass to Stage 2")

    print("\n## Score Histogram")
    max_bar = max(score_histogram.values()) if score_histogram else 1
    for bucket in sorted(score_histogram):
        cnt = score_histogram[bucket]
        bar = "█" * max(1, cnt * 60 // max_bar)
        print(f"  {bucket:>3}-{bucket + 4:<3} | {bar} {cnt:>7,}")

    print("\n## Signal Hit Rates")
    for sig, cnt in signal_hits.most_common(25):
        print(f"  {sig:25s}: {cnt:>8,} ({cnt / total * 100:5.2f}%)")

    print("\n## Source × Level Breakdown")
    header = f"  {'Source':<20s} {'Total':>8s} {'Critical':>10s} {'High':>10s} {'Normal':>10s} {'Low':>10s} {'%Stage2':>8s}"
    print(header)
    for src in sorted(source_level, key=lambda s: sum(source_level[s].values()), reverse=True):
        c = source_level[src]
        t = sum(c.values())
        s2_pct = (c["critical"] + c["high"]) / t * 100 if t else 0
        print(
            f"  {src:<20s} {t:>8,} {c['critical']:>10,} {c['high']:>10,} {c['normal']:>10,} {c['low']:>10,} {s2_pct:>7.1f}%"
        )

    for label, examples in [
        (f"CRITICAL (score >= {CRITICAL_THRESHOLD})", critical_examples[:15]),
        (f"HIGH (score {HIGH_THRESHOLD}-{CRITICAL_THRESHOLD - 1})", high_examples[:15]),
        ("BORDERLINE MISSES (score 25-31)", borderline_miss[:15]),
    ]:
        print(f"\n## {label}")
        for score, src, text, reasons in examples:
            print(f"  [{score:>3}] @{src:<18s} {text}")
            print(f"        {', '.join(reasons[:6])}")


if __name__ == "__main__":
    main()
