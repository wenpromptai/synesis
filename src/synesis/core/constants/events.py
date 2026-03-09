"""Event radar, 13F monitoring, and digest timing constants."""

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

# Digest timing
DIGEST_WHATS_COMING_DAYS = 7

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
