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
FINNHUB_CACHE_TTL_SYMBOL = 604800  # 7 days
FINNHUB_CACHE_TTL_US_SYMBOLS = 86400  # 24 hours for bulk US symbol list

# SEC EDGAR cache TTLs
SEC_EDGAR_CACHE_TTL_SUBMISSIONS = 3600  # 1 hour for company submissions/filings
SEC_EDGAR_CACHE_TTL_CIK_MAP = 86400  # 24 hours for ticker→CIK mapping

# NASDAQ cache TTLs
NASDAQ_CACHE_TTL_EARNINGS = 21600  # 6 hours for earnings calendar

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
KALSHI_EVENT_FETCH_CONCURRENCY = 5  # Max concurrent event category fetches
KALSHI_EVENT_FETCH_DELAY = 0.15  # Seconds between event fetches (rate pacing)
KALSHI_CATEGORY_CACHE_TTL = 3600  # 1 hour TTL for event category cache

# Cross-platform arbitrage (Feature 1)
CROSS_PLATFORM_ARB_MIN_GAP = 0.03  # 3 cents min price gap (must exceed ~2% comms)
CROSS_PLATFORM_MATCH_SIMILARITY = 0.80  # Cosine similarity threshold
ARB_ALERT_COOLDOWN_MINUTES = 10
PRICE_UPDATE_CHANNEL = "synesis:mkt_intel:price_update"

# Wallet Tracker
WALLET_API_DELAY_SECONDS = 0.1  # Rate limiting delay between wallet API calls
WALLET_SCORE_CACHE_TTL = 3600  # 1 hour cache for wallet scores in Redis
WALLET_RESCORE_INTERVAL_SECONDS = 86400  # 24h between full re-score cycles
WALLET_DISCOVERY_TOP_N_MARKETS = 15  # Top markets by volume to scan for wallets
WALLET_TOP_HOLDERS_LIMIT = 10  # Top holders to fetch per market
WALLET_ACTIVITY_MAX_MARKETS = 20  # Max markets to check for wallet activity

# Fast-track auto-watch thresholds
FAST_TRACK_MAX_WASH_RATIO = 0.30  # Hard filter: reject if wash_trade_ratio exceeds this
CONSISTENT_INSIDER_MIN_WIN_RATE = 0.50  # Consistent Insider: minimum win rate
CONSISTENT_INSIDER_MIN_PNL_PER_POSITION = 10_000  # Consistent Insider: min PnL per position (USDC)
FRESH_INSIDER_MIN_POSITION = 50_000  # Fresh Insider: min open position size (USDC)

# ─────────────────────────────────────────────────────────────
# Watchlist Intelligence (Flow 4)
# ─────────────────────────────────────────────────────────────
WATCHLIST_INTEL_REDIS_PREFIX = "synesis:watchlist_intel"
WATCHLIST_INTEL_DEFAULT_BATCH_SIZE = 5  # parallel ticker fetches
