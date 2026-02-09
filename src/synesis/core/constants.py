"""Application-wide constants.

These are fixed values that don't change between environments.
For configurable values, see config.py Settings.
"""

# ─────────────────────────────────────────────────────────────
# API Rate Limits (external constraints)
# ─────────────────────────────────────────────────────────────
FINNHUB_RATE_LIMIT_CALLS_PER_MINUTE = 60  # Free tier limit
FINNHUB_WS_MAX_SYMBOLS = 50  # Free tier WebSocket subscription limit

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
DEFAULT_SIGNALS_OUTPUT_DIR = "output/signals"

# ─────────────────────────────────────────────────────────────
# Prediction Market API URLs
# ─────────────────────────────────────────────────────────────
DEFAULT_POLYMARKET_DATA_API_URL = "https://data-api.polymarket.com"
DEFAULT_POLYMARKET_CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# ─────────────────────────────────────────────────────────────
# Market Intelligence (Flow 3)
# ─────────────────────────────────────────────────────────────
MARKET_INTEL_SNAPSHOT_INTERVAL = 300  # 5 min snapshots for volume tracking
MARKET_INTEL_MAX_TRACKED_MARKETS = 100  # Max markets to track via WebSocket
MARKET_INTEL_REDIS_PREFIX = "synesis:mkt_intel"
KALSHI_RATE_LIMIT_CALLS_PER_SECOND = 10  # Kalshi public API rate limit
