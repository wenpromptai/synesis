"""Display constants — Discord colors, emoji/icon maps, digest formatting."""

# ─────────────────────────────────────────────────────────────
# Discord Embed Colors (decimal integers)
# ─────────────────────────────────────────────────────────────
COLOR_HEADER = 0x2B2D31
COLOR_CALENDAR = 0x5865F2  # Blurple

COLOR_BULLISH = 0x57F287  # Green
COLOR_BEARISH = 0xED4245  # Red
COLOR_NEUTRAL = 0xFEE75C  # Yellow
COLOR_URGENT = 0xE67E22  # Orange
COLOR_CRITICAL = 0xED4245  # Red (same as COLOR_BEARISH — intentional)

SECTOR_COLORS: dict[str, int] = {
    "macro": 0x5865F2,
    "tech": 0x57F287,
    "corporate": 0x3498DB,
    "sector": 0xE67E22,
    "regulatory": 0x9B59B6,
    "earnings": 0xFEE75C,
    "13f_filing": 0x1ABC9C,
    "fed": 0x5865F2,
}

# ─────────────────────────────────────────────────────────────
# Emoji / Icon Maps
# ─────────────────────────────────────────────────────────────
THEME_EMOJI: dict[str, str] = {
    "macro": "\U0001f30d",
    "tech": "\U0001f680",
    "corporate": "\U0001f3e2",
    "sector": "\U0001f3ed",
    "regulatory": "\u2696\ufe0f",
    "earnings": "\U0001f4b0",
    "13f_filing": "\U0001f3e6",
    "fed": "\U0001f3db\ufe0f",
}

SENTIMENT_ICON: dict[str, str] = {
    "bullish": "\U0001f7e2",
    "bearish": "\U0001f534",
    "neutral": "\u26aa",
}

CATEGORY_EMOJI: dict[str, str] = {
    "fed": "\U0001f3db\ufe0f",
    "economic_data": "\U0001f4ca",
    "release": "\U0001f680",
    "conference": "\U0001f399\ufe0f",
    "earnings": "\U0001f4b0",
    "regulatory": "\u2696\ufe0f",
    "13f_filing": "\U0001f3e6",
}

DIRECTION_ICON: dict[str, str] = {
    "long": "\U0001f4c8",
    "short": "\U0001f4c9",
    "hedge": "\U0001f6e1\ufe0f",
    "watch": "\U0001f440",
}

SOURCE_LABEL: dict[str, str] = {
    "calendar": "\U0001f4cb Calendar",
    "surprise": "\u26a1 Surprise",
    "analysis": "\U0001f4ca Analysis",
}

# ─────────────────────────────────────────────────────────────
# Digest State Tracking
# ─────────────────────────────────────────────────────────────
LAST_DIGEST_KEY = "synesis:event_radar:last_digest"
LAST_DIGEST_TTL = 172800  # 48 hours
