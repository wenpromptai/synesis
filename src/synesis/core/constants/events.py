"""Event radar, 13F monitoring, surprise detection, and digest timing constants."""

# ─────────────────────────────────────────────────────────────
# Event Radar
# ─────────────────────────────────────────────────────────────
EVENT_RADAR_STRUCTURED_DAYS_AHEAD = 30
EVENT_RADAR_EARNINGS_DAYS_AHEAD = 14
EVENT_RADAR_DIGEST_DAYS = 7  # Default digest window (days)
EVENT_RADAR_DIGEST_COMING_UP_DAYS = 14  # "Coming up" section lookahead

# ─────────────────────────────────────────────────────────────
# SEC 13F Hedge Fund Monitoring
# Fund registry lives in processing/events/sources.yaml
# ─────────────────────────────────────────────────────────────
SEC_13F_HOLDINGS_CACHE_TTL = 604800  # 7 days (filings don't change)
SEC_13F_DIFF_CACHE_TTL = 86400  # 24 hours
SEC_13F_SEEN_TTL = 7776000  # 90 days

# 13F filing deadlines (quarter_end_month -> deadline month, day)
SEC_13F_DEADLINES: dict[int, tuple[int, int]] = {
    12: (2, 14),  # Q4 (Dec 31) -> Feb 14
    3: (5, 15),  # Q1 (Mar 31) -> May 15
    6: (8, 14),  # Q2 (Jun 30) -> Aug 14
    9: (11, 14),  # Q3 (Sep 30) -> Nov 14
}

# Surprise event detection
SURPRISE_SEARCH_QUERIES: list[str] = [
    "breaking market moving news {date_range}",
    "major corporate announcement {date_range}",
    "central bank surprise decision {date_range}",
    "new AI model released launched today {date_range}",
    "new LLM foundation model open source release {date_range}",
    "new technology breakthrough product launch announced {date_range}",
    "major investment deal billion dollar announcement {date_range}",
    "major partnership collaboration strategic alliance announced {date_range}",
    "new business venture startup funding raised {date_range}",
    "geopolitical conflict commodity oil impact {date_range}",
    "major acquisition merger {date_range}",
    "regulatory enforcement antitrust ruling {date_range}",
    "earnings surprise beat miss guidance {date_range}",
    "economic data surprise inflation jobs GDP {date_range}",
]
SURPRISE_MAX_RESULTS = 15

EXTRACTOR_MAX_CONTENT_CHARS = 10000

# Digest timing
DIGEST_YESTERDAY_LOOKBACK_HOURS = 28  # Slightly more than 24h to avoid gaps
DIGEST_WHATS_COMING_DAYS = 7
DIGEST_MIN_EARNINGS_IMPORTANCE = 6

# ─────────────────────────────────────────────────────────────
# FRED series mapping for economic_data enrichment
# Maps event title keywords → (series_id, units) for FRED observation lookup
# ─────────────────────────────────────────────────────────────
FRED_OUTCOME_SERIES: dict[str, tuple[str, str]] = {
    "CPI": ("CPIAUCSL", "pc1"),  # YoY % change
    "Core CPI": ("CPILFESL", "pc1"),
    "GDP": ("GDP", "pch"),  # QoQ % change
    "NFP": ("PAYEMS", "chg"),  # Monthly change (thousands)
    "Non-Farm Payroll": ("PAYEMS", "chg"),
    "PCE": ("PCEPI", "pc1"),  # YoY % change
    "Core PCE": ("PCEPILFE", "pc1"),
    "PPI": ("PPIACO", "pch"),  # MoM % change
    "Personal Income": ("PI", "pch"),
    "Retail Sales": ("RSAFS", "pch"),
    "Unemployment": ("UNRATE", "lin"),  # Level (%)
}
