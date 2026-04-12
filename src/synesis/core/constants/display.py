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

# ─────────────────────────────────────────────────────────────
# Emoji / Icon Maps
# ─────────────────────────────────────────────────────────────
CATEGORY_EMOJI: dict[str, str] = {
    "fed": "\U0001f3db\ufe0f",
    "economic_data": "\U0001f4ca",
    "release": "\U0001f680",
    "conference": "\U0001f399\ufe0f",
    "earnings": "\U0001f4b0",
    "regulatory": "\u2696\ufe0f",
    "13f_filing": "\U0001f3e6",
}

# ─────────────────────────────────────────────────────────────
# Digest State Tracking
# ─────────────────────────────────────────────────────────────
LAST_DIGEST_KEY = "synesis:event_radar:last_digest"
LAST_DIGEST_TTL = 172800  # 48 hours
