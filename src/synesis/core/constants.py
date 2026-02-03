"""Application-wide constants.

These are fixed values that don't change between environments.
For configurable values, see config.py Settings.
"""

# ─────────────────────────────────────────────────────────────
# API Rate Limits (external constraints)
# ─────────────────────────────────────────────────────────────
FINNHUB_RATE_LIMIT_CALLS_PER_MINUTE = 60  # Free tier limit

# ─────────────────────────────────────────────────────────────
# Message Limits (platform constraints)
# ─────────────────────────────────────────────────────────────
TELEGRAM_MAX_MESSAGE_LENGTH = 4096  # Telegram API limit

# ─────────────────────────────────────────────────────────────
# Cache TTLs (sensible defaults)
# ─────────────────────────────────────────────────────────────
PRICE_CACHE_TTL_SECONDS = 1800  # 30 minutes
DEDUP_CACHE_TTL_SECONDS = 3600  # 60 minutes
FINNHUB_CACHE_TTL_FINANCIALS = 3600  # 1 hour
FINNHUB_CACHE_TTL_INSIDER = 21600  # 6 hours
FINNHUB_CACHE_TTL_EARNINGS = 86400  # 24 hours
FINNHUB_CACHE_TTL_FILINGS = 21600  # 6 hours
FINNHUB_CACHE_TTL_SYMBOL = 604800  # 7 days
FINNHUB_CACHE_TTL_US_SYMBOLS = 86400  # 24 hours for bulk US symbol list

# ─────────────────────────────────────────────────────────────
# Default Thresholds (can be overridden in Settings)
# ─────────────────────────────────────────────────────────────
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_TICKER_RELEVANCE_THRESHOLD = 0.6
DEFAULT_CONVICTION_THRESHOLD = 0.7

# ─────────────────────────────────────────────────────────────
# Processing Limits
# ─────────────────────────────────────────────────────────────
MAX_POSTS_FOR_LLM_ANALYSIS = 30
MAX_TICKERS_DISPLAY = 50
MAX_MARKETS_FOR_ANALYSIS = 7

# ─────────────────────────────────────────────────────────────
# API URL Defaults (used as defaults in Settings)
# ─────────────────────────────────────────────────────────────
DEFAULT_POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com"
DEFAULT_FINNHUB_WS_URL = "wss://ws.finnhub.io"
DEFAULT_FINNHUB_API_URL = "https://finnhub.io/api/v1"
DEFAULT_SIGNALS_OUTPUT_DIR = "shared/output"
