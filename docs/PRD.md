# PRD2: Signal Generation System (Flows 1-3)

**Status:** Ready for Implementation
**Created:** 2026-01-28
**Scope:** Flows 1-3 signal generation only

---

## Overview

Three independent signal-generating flows, each producing actionable outputs on their own schedule.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL GENERATION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FLOW 1: Breaking News         FLOW 2: Sentiment            FLOW 3:        │
│  ─────────────────────         ────────────────             Prediction     │
│                                                             Market Monitor │
│  Twitter Stream ───┐           RSS.app (Reddit) ──┐         ────────────── │
│                    ├─► Dedupe  r/wsb, r/stocks    │                        │
│  Telegram Stream ──┘      │           │           │         Polymarket     │
│  (@marketfeed,            ▼           ▼           │         Kalshi         │
│   @disclosetv)          LLM    ┌─────────────┐    │              │         │
│                          │     │ 1. Discover │    │              ▼         │
│                          ▼     │    tickers  │    │         Every 15min    │
│                       Signal:  │ 2. Add to   │    │         Signal:        │
│                       • Impact │    watchlist│    │         • Wallets      │
│                       • PM opp └──────┬──────┘    │         • Volume       │
│                          │            │           │         • Entry opps   │
│                          │            ▼           │                        │
│                          │      ┌───────────┐     │                        │
│                          └─────►│ WATCHLIST │◄────┘                        │
│                                 └─────┬─────┘                              │
│                                       │                                    │
│                                       ▼                                    │
│                          ┌────────────────────────┐                        │
│                          │ Monitor sentiment via  │                        │
│                          │ Twitter Search + Reddit│                        │
│                          └───────────┬────────────┘                        │
│                                      │                                     │
│                                      ▼                                     │
│                               Every 6hr Signal:                            │
│                               • Watchlist + changes                        │
│                               • Sentiment per ticker                       │
│                               • Evidence (posts/tweets)                    │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Flow 1: News & Analysis Intelligence

### Purpose
Process breaking news and analysis from curated sources:
- **News** (high urgency): Detect market-moving events → find prediction market arbitrage before odds adjust
- **Analysis** (normal urgency): Extract insights → identify tickers/sectors for monitoring

Both feed the same pipeline; urgency is tagged by source type.

### Data Sources (Already Available)

Both **news** and **analysis** accounts feed into the same pipeline. Urgency is determined by source type, not LLM.

| Source | Type | Urgency | Accounts/Channels | Status |
|--------|------|---------|-------------------|--------|
| **Twitter** | News | High | @DeItaone, @FinancialJuice | ✅ Ready |
| **Twitter** | Analysis | Normal | (configured separately) | ✅ Ready |
| **Telegram** | News | High | @marketfeed, @disclosetv | ✅ Ready |
| **Telegram** | Analysis | Normal | (configured separately) | ✅ Ready |

Both streams feed into the **same unified pipeline** — deduplicate first, then LLM classify. **Urgency is tagged based on source configuration, not LLM output.**

### Pipeline

```
Twitter Stream ─────────┐
  News: @DeItaone, etc. │ → tagged: HIGH urgency
  Analysis: @spotgamma  │ → tagged: NORMAL urgency
                        │
                        ├──► Unified Queue ──► Deduplicate ──► LLM Classify
                        │    (Redis Streams)   (SemHash)       (PydanticAI)
Telegram Stream ────────┘          │                               │
  News: @marketfeed,               │                               │
        @disclosetv                │                               ▼
  Analysis: (as configured)  (duplicates               ┌───────────────────┐
                              logged)                  │      OUTPUT       │
                                                       ├───────────────────┤
                                                       │ • Signal          │
                                                       │   + urgency (from │
                                                       │     source type)  │
                                                       │ • PM Search       │
                                                       │ • Watchlist       │
                                                       └───────────────────┘
```

**Source tagging**: Each message tagged with `source_type` (news/analysis) at ingestion. This determines urgency.

**Deduplication**: SemHash with Model2Vec embeddings, <5ms per message, 0.85 similarity threshold. First source wins, duplicates logged for coverage tracking.

### LLM Classification Schema

```python
from pydantic import BaseModel
from enum import Enum

class EventType(str, Enum):
    macro = "macro"           # Fed, CPI, GDP
    earnings = "earnings"     # Company results
    geopolitical = "geopolitical"  # Wars, sanctions
    corporate = "corporate"   # M&A, CEO changes
    regulatory = "regulatory" # SEC, antitrust
    crypto = "crypto"         # ETF, exchange news

class ImpactLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"

class Direction(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"

class BreakingClassification(BaseModel):
    """LLM extracts all of this from a single news/analysis item."""

    # Event classification
    event_type: EventType
    summary: str
    confidence: float

    # Impact assessment (LLM determines this)
    predicted_impact: ImpactLevel
    market_direction: Direction

    # Extracted entities (for Flow 2 watchlist)
    tickers: list[str]           # $AAPL, $TSLA
    sectors: list[str]           # "semiconductors", "energy"

    # Prediction market relevance
    prediction_market_relevant: bool
    search_keywords: list[str]   # Keywords to search Polymarket/Kalshi
    related_markets: list[str]   # e.g., "Fed rate cut", "Trump tariffs"

# NOTE: Urgency is NOT determined by LLM - it comes from source configuration
# News sources (DeItaone, marketfeed, etc.) → high urgency
# Analysis sources → normal urgency
```

### Prediction Market Search

After LLM classification, search Polymarket and Kalshi for related markets:

```python
from synesis.integrations.polymarket import PolymarketClient
from synesis.integrations.kalshi import KalshiClient

async def find_arbitrage_opportunities(
    classification: BreakingClassification
) -> list[MarketOpportunity]:
    """
    Search prediction markets for entry opportunities.
    Key: Find markets where odds haven't moved yet.
    """
    poly = PolymarketClient()
    kalshi = KalshiClient()

    opportunities = []

    for keyword_set in classification.search_keywords:
        # Search both platforms
        poly_markets = await poly.search_markets(keyword_set)
        kalshi_markets = await kalshi.search_markets(keyword_set)

        for market in poly_markets + kalshi_markets:
            # Check if odds suggest market hasn't priced in news yet
            if classification.predicted_impact == ImpactLevel.high:
                if classification.market_direction == Direction.bullish:
                    # News is bullish but market prob is low = opportunity
                    if market.yes_price < 0.6:
                        opportunities.append(MarketOpportunity(
                            market=market,
                            direction="yes",
                            reason=f"Breaking news bullish, market at {market.yes_price:.0%}",
                            news_summary=classification.summary
                        ))
                # ... similar for bearish

    return opportunities
```

### Output Signal

```python
class SourceType(str, Enum):
    news = "news"         # High urgency - act fast
    analysis = "analysis" # Normal urgency - consider

class NewsSignal(BaseModel):
    """Real-time signal emitted for each news/analysis item."""

    timestamp: datetime

    # Source info (urgency derived from source_type)
    source_platform: str      # "twitter" or "telegram"
    source_account: str       # "@DeItaone", "@marketfeed", etc.
    source_type: SourceType   # From config - determines urgency
    raw_text: str

    # Urgency (derived from source_type, not LLM)
    @property
    def urgency(self) -> str:
        return "high" if self.source_type == SourceType.news else "normal"

    # LLM Classification
    classification: BreakingClassification

    # Prediction market opportunities (if any found)
    opportunities: list[MarketOpportunity]

    # Watchlist additions (sent to Flow 2)
    watchlist_tickers: list[str]
    watchlist_sectors: list[str]
```

### Signal Delivery
- **Real-time** (as news arrives)
- Telegram bot notification for high-impact opportunities
- Log all signals to database for backtesting

---

## Flow 2: Sentiment Intelligence

### Purpose
1. **Discover tickers** from Reddit discussion (WSB, stocks, options)
2. **Monitor sentiment** for all watchlist tickers (from Flow 1 + Reddit discoveries)
3. **Generate signals** with sentiment changes and supporting evidence

### Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   INPUT                                                         │
│   ─────                                                         │
│   Flow 1 extracts tickers → Auto-added to watchlist (Redis)     │
│   Reddit discovers tickers → Also added to watchlist            │
│   • 7-day TTL per ticker (auto-expires if no new mentions)      │
│                                                                 │
│                              ↓                                  │
│                                                                 │
│   DATA COLLECTION                                               │
│   ───────────────                                               │
│                                                                 │
│   Twitter ─────┐                                                │
│   • TwitterAPI.io search for $TICKER                            │
│                │                                                │
│                ├──► Store in Redis by ticker                    │
│                │                                                │
│   Reddit ──────┘                                                │
│   • r/WSB, r/stocks, r/options via RSS.app                      │
│                                                                 │
│                              ↓                                  │
│                                                                 │
│   ANALYSIS (LLM per post)                                       │
│   ───────────────────────                                       │
│                                                                 │
│   1. SENTIMENT CLASSIFIER                                       │
│      • Classify each post: bullish/bearish/neutral              │
│      • Tag emotion: fomo, panic, euphoria, etc.                 │
│      • Aggregate per ticker → score (-1.0 to +1.0)              │
│                                                                 │
│   2. CHANGE DETECTOR                                            │
│      • Compare vs last 6h signal                                │
│      • Flag extremes (>85% one direction)                       │
│      • Detect volume spikes (zscore > 2)                        │
│                                                                 │
│                              ↓                                  │
│                                                                 │
│   OUTPUT (Every 6 Hours)                                        │
│   ──────────────────────                                        │
│   • Watchlist + changes (added/removed tickers)                 │
│   • Sentiment per ticker with evidence (top posts)              │
│   • Extreme readings & biggest movers                           │
│   • LLM narrative summary                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Redis Keys:**

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `watchlist:tickers` | Set | None | Active tickers to monitor |
| `watchlist:ttl:{ticker}` | String | 7 days | Auto-expires ticker |

**Example Flow:**

```
Flow 1: "@DeItaone: KIOXIA IPO priced at $15"
           ↓
        LLM extracts ticker: KIOXIA
           ↓
        Redis: SADD watchlist:tickers KIOXIA
           ↓
Flow 2: Now monitoring KIOXIA sentiment
           ↓
        Twitter search: "$KIOXIA"
        Reddit scan: mentions of KIOXIA
           ↓
        Every 6h signal:
        • KIOXIA: +0.6 bullish, 45 mentions
        • Dominant emotion: "optimistic"
        • Top post: "KIOXIA looks solid at IPO price..."
```

---

### Data Sources

| Source | Implementation | Purpose | Status |
|--------|---------------|---------|--------|
| **Reddit** | RSS.app | Ticker discovery + sentiment | ⏳ Pending account |
| **Twitter Search** | TwitterAPI.io | Sentiment coverage | ✅ Ready |

### Watchlist Sources (Two Inputs)

```
Flow 1 (Breaking News) ────► extracts tickers ────┐
                                                  ├───► WATCHLIST
Reddit (WSB, stocks, etc) ─► discovers tickers ───┘
                                                        │
                                                        ▼
                                              Monitor sentiment via
                                              Twitter Search + Reddit
```

- **From Flow 1**: Tickers/sectors extracted from breaking news
- **From Reddit**: LLM analyzes posts to find valuable tickers being discussed
- **Manual additions**: Can add tickers directly
- **TTL**: 7 days (auto-expire unless renewed by new mentions)

### Reddit via RSS.app

RSS.app provides clean RSS feeds for subreddits without Reddit API authentication hassles.

**Target Subreddits:**
| Subreddit | RSS URL Pattern | Poll Interval |
|-----------|-----------------|---------------|
| r/wallstreetbets | `https://rss.app/feeds/...` | 5 min |
| r/stocks | `https://rss.app/feeds/...` | 15 min |
| r/options | `https://rss.app/feeds/...` | 15 min |
| r/investing | `https://rss.app/feeds/...` | 30 min |

**Setup Steps:**
1. Register at rss.app
2. Create RSS feeds for each subreddit (new posts)
3. Store feed URLs in config

```python
import feedparser
from datetime import datetime

class RedditRSSClient:
    """Reddit via RSS.app - no API key needed after setup."""

    def __init__(self, feeds: dict[str, str]):
        """
        Args:
            feeds: {"wallstreetbets": "https://rss.app/feeds/xxx", ...}
        """
        self.feeds = feeds

    async def fetch_subreddit(self, subreddit: str) -> list[RedditPost]:
        url = self.feeds[subreddit]
        feed = feedparser.parse(url)

        posts = []
        for entry in feed.entries:
            posts.append(RedditPost(
                id=entry.id,
                subreddit=subreddit,
                title=entry.title,
                content=entry.get("summary", ""),
                url=entry.link,
                published=entry.get("published"),
                source="reddit"
            ))
        return posts
```

### Pipeline

```
Every 6 hours:

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: TICKER DISCOVERY                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Fetch Reddit posts (RSS.app)                                           │
│  from: r/wallstreetbets, r/stocks, r/options                            │
│         │                                                               │
│         ▼                                                               │
│  LLM: Analyze posts                                                     │
│  - Extract mentioned tickers                                            │
│  - Assess why ticker is being discussed                                 │
│  - Score relevance (is this worth watching?)                            │
│         │                                                               │
│         ▼                                                               │
│  Add valuable tickers to WATCHLIST                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SENTIMENT MONITORING                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOR EACH ticker in WATCHLIST (Flow 1 + Reddit discoveries):            │
│         │                                                               │
│         ├─► Fetch Twitter posts (TwitterAPI.io search)                  │
│         │                                                               │
│         ├─► Use already-fetched Reddit posts                            │
│         │                                                               │
│         └─► LLM: Classify sentiment per post                            │
│                  │                                                      │
│                  ▼                                                      │
│             AGGREGATE per ticker:                                       │
│             - bullish/bearish/neutral ratios                            │
│             - dominant emotion                                          │
│             - volume vs historical                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SIGNAL GENERATION                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - Detect sentiment changes vs last signal                              │
│  - Flag extremes (>85% one direction)                                   │
│  - Generate narrative summary                                           │
│  - Emit signal with evidence                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### LLM Ticker Discovery (Step 1)

```python
class TickerMention(BaseModel):
    """LLM extracts tickers from Reddit posts and assesses value."""
    ticker: str
    company_name: str | None
    mention_context: str  # Why is this being discussed?
    catalyst: str | None  # Earnings, news, technical setup, etc.
    sentiment_hint: Literal["bullish", "bearish", "neutral", "mixed"]
    worth_watching: bool  # Should this go to watchlist?
    reasoning: str        # Why worth watching (or not)

class RedditPostAnalysis(BaseModel):
    """LLM analysis of a single Reddit post."""
    post_id: str
    subreddit: str
    tickers_mentioned: list[TickerMention]
    post_quality: Literal["high", "medium", "low", "spam"]
    is_dd: bool           # Is this due diligence / research?
    key_insight: str | None
```

### LLM Sentiment Classification (Step 2)

```python
class StockEmotion(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    optimistic = "optimistic"
    pessimistic = "pessimistic"
    fomo = "fomo"           # Crowded trade warning
    panic = "panic"         # Potential bottom
    euphoria = "euphoria"   # Top warning
    despair = "despair"     # Potential bottom
    neutral = "neutral"
    confused = "confused"
    skeptical = "skeptical"
    hopeful = "hopeful"

class PostSentiment(BaseModel):
    """LLM classifies each post."""
    post_id: str
    ticker: str
    emotion: StockEmotion
    polarity: Literal["bullish", "bearish", "neutral"]
    intensity: float  # 0-1
    key_quote: str    # Most relevant excerpt
    reasoning: str
```

### Output Signal (Every 6 Hours)

```python
class TickerSentimentSummary(BaseModel):
    ticker: str

    # Aggregated sentiment
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    dominant_emotion: StockEmotion

    # Change metrics
    sentiment_delta_6h: float  # vs last signal
    volume_zscore: float       # mention volume vs 30-day avg

    # Flags
    is_extreme_bullish: bool   # >85% bullish
    is_extreme_bearish: bool   # >85% bearish
    is_volume_spike: bool      # zscore > 2

    # Evidence
    top_posts: list[PostSentiment]  # Top 5 most influential posts

class SentimentSignal(BaseModel):
    """Emitted every 6 hours."""

    timestamp: datetime
    signal_period: str  # "6h"

    # Current watchlist
    watchlist: list[str]
    watchlist_added: list[str]    # New since last signal
    watchlist_removed: list[str]  # Expired/removed

    # Per-ticker sentiment
    ticker_sentiments: list[TickerSentimentSummary]

    # Highlights
    extreme_sentiments: list[str]  # Tickers with extreme readings
    biggest_movers: list[str]      # Largest sentiment changes

    # Summary (LLM-generated)
    narrative_summary: str  # "NVDA sentiment shifted bearish on DeepSeek news..."
```

### Signal Delivery
- **Every 6 hours** (configurable)
- Telegram bot with summary
- Full signal logged to database

---

## Flow 3: Prediction Market Intelligence

### Purpose
Monitor Polymarket/Kalshi for smart money activity → generate periodic signals on trading opportunities.

### Data Sources

| Source | API | Status |
|--------|-----|--------|
| **Polymarket** | Gamma API (read-only, no auth) | ✅ Ready |
| **Kalshi** | Trade API v2 (read-only, no auth) | ✅ Ready |

### Monitoring Targets

1. **Profitable Wallets**: Track wallets with high historical accuracy
2. **Volume Spikes**: Unusual activity suggesting informed trading
3. **Odds Movements**: Significant probability changes
4. **Expiring Markets**: High-urgency opportunities

### Pipeline (Every 15 Minutes)

```
Every 15 minutes:

1. SCAN markets
   │
   ├─► Get trending markets (by volume)
   ├─► Get expiring markets (<24h)
   └─► Detect volume spikes (zscore > 2)

2. FOR EACH interesting market:
   │
   ├─► Fetch recent trades
   ├─► Check for known profitable wallets
   └─► Calculate odds movement

3. SCORE opportunities:
   │
   ├─► Insider score (wallet history)
   ├─► Volume spike score
   └─► Urgency score (time to expiration)

4. GENERATE signal
```

### Wallet Tracking

```python
class WalletProfile(BaseModel):
    """Track wallet performance over time."""
    address: str

    # Historical performance
    total_trades: int
    profitable_trades: int
    win_rate: float

    # Pre-news accuracy (key insider signal)
    pre_news_trades: int      # Trades within 1hr before major news
    pre_news_accuracy: float  # Win rate on pre-news trades

    # Calculated score
    insider_score: float  # 0-1, likelihood of informed trading

    @property
    def is_profitable_wallet(self) -> bool:
        return (
            self.total_trades >= 10 and
            self.win_rate > 0.6 and
            self.insider_score > 0.5
        )
```

### Volume Spike Detection

```python
async def detect_volume_spike(
    market: SimpleMarket,
    threshold_zscore: float = 2.0
) -> VolumeAlert | None:
    """
    Detect unusual volume that may indicate informed trading.
    """
    # Get historical volume
    history = await get_market_volume_history(market.id, days=30)

    avg_hourly = statistics.mean(history)
    std_hourly = statistics.stdev(history)

    current_hourly = market.volume_24h / 24
    zscore = (current_hourly - avg_hourly) / std_hourly

    if zscore > threshold_zscore:
        return VolumeAlert(
            market_id=market.id,
            market_question=market.question,
            zscore=zscore,
            current_volume=market.volume_24h,
            avg_volume=avg_hourly * 24,
            alert_type="volume_spike"
        )

    return None
```

### Output Signal (Every 15 Minutes)

```python
class MarketOpportunity(BaseModel):
    """A potential trading opportunity."""
    market_id: str
    platform: Literal["polymarket", "kalshi"]
    question: str
    current_prob: float

    # Opportunity details
    suggested_direction: Literal["yes", "no"]
    confidence: float

    # Why this is interesting
    triggers: list[str]  # ["volume_spike", "insider_activity", "expiring_soon"]

    # Evidence
    volume_24h: float
    volume_zscore: float | None
    insider_wallets_active: list[str]
    hours_to_expiration: float | None

class Flow3Signal(BaseModel):
    """Emitted every 15 minutes."""

    timestamp: datetime
    signal_period: str  # "15min"

    # Market overview
    total_markets_scanned: int
    trending_markets: list[SimpleMarket]
    expiring_soon: list[SimpleMarket]  # <24h

    # Alerts
    volume_spikes: list[VolumeAlert]
    insider_activity: list[InsiderAlert]
    odds_movements: list[OddsMovement]

    # Opportunities (ranked)
    opportunities: list[MarketOpportunity]

    # Summary
    market_uncertainty_index: float  # Avg entropy across markets
    informed_activity_level: float   # Overall insider signal strength
```

### Signal Delivery
- **Every 15 minutes** (respecting rate limits)
- Telegram bot for high-confidence opportunities
- Dashboard-ready data structure

### Wallet Discovery & Tracking

#### How to Find Profitable Wallets

**Phase 1: Historical Backfill (One-time)**
```python
async def discover_profitable_wallets():
    """
    Analyze resolved markets to find consistently winning wallets.

    Source: Polymarket CLOB API
    - GET /markets (filter: closed=true)
    - GET /get_market_trades_events/{condition_id}
    """
    # 1. Fetch all resolved markets (last 6 months)
    markets = await polymarket.get_resolved_markets(since=six_months_ago)

    # 2. For each market, get all trades
    for market in markets:
        trades = await polymarket.get_market_trades(market.condition_id)

        # 3. Calculate wallet performance
        for trade in trades:
            await update_wallet_metrics(
                wallet=trade.maker_address,
                market=market,
                trade=trade
            )

    # 4. Identify top performers
    return await db.fetch("""
        SELECT * FROM wallet_metrics
        WHERE total_trades >= 10
          AND win_rate > 0.55
          AND insider_score > 0.5
        ORDER BY insider_score DESC
    """)
```

**Phase 2: Real-Time Monitoring (Continuous)**
```python
# WebSocket subscription for live trades
ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

async def monitor_wallets():
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({
            "action": "subscribe",
            "subscriptions": [{"topic": "activity", "type": "trades"}]
        }))

        async for msg in ws:
            trade = json.loads(msg)
            wallet = trade.get("proxyWallet")

            # Check if watched wallet
            if await redis.sismember("wallets:watched", wallet):
                await emit_alert(trade)
```

#### Insider Score Calculation

```python
def calculate_insider_score(metrics: WalletMetrics) -> float:
    """
    Composite score based on multiple signals.

    Research basis: 86% of Polymarket accounts have negative P&L,
    only 0.51% profit >$1K. High win rates are rare and suspicious.

    Sources: PolyTrack, Polywhaler research
    """
    # Component weights
    weights = {
        "pre_news_accuracy": 0.40,  # Most important signal
        "pre_news_frequency": 0.20,
        "timing_consistency": 0.15,
        "conviction_score": 0.15,
        "account_freshness": 0.10,
    }

    # Pre-news accuracy (trades within 1hr before resolution)
    pre_news_accuracy = metrics.pre_news_wins / max(metrics.pre_news_trades, 1)

    # Pre-news frequency (ratio of pre-news trades to total)
    pre_news_frequency = metrics.pre_news_trades / max(metrics.total_trades, 1)

    # Account freshness (newer = more suspicious)
    days_active = (now() - metrics.first_seen_at).days
    account_freshness = 1.0 if days_active < 7 else 0.4 if days_active < 30 else 0.1

    return (
        pre_news_accuracy * weights["pre_news_accuracy"] +
        pre_news_frequency * weights["pre_news_frequency"] +
        # ... other components
    )
```

#### Pre-News Trade Detection

```sql
-- Find trades within 1 hour before market resolution
SELECT
    t.wallet_address,
    t.market_id,
    t.direction,
    t.size,
    t.traded_at,
    m.resolved_at,
    EXTRACT(EPOCH FROM (m.resolved_at - t.traded_at)) / 60 AS minutes_before,
    CASE
        WHEN t.direction = m.winning_outcome THEN TRUE
        ELSE FALSE
    END AS won
FROM wallet_trades t
JOIN markets m ON t.market_id = m.id
WHERE m.resolved_at IS NOT NULL
  AND t.traded_at BETWEEN m.resolved_at - INTERVAL '1 hour' AND m.resolved_at;
```

#### Warning Signs (Insider Detection)

| Signal | Threshold | Action |
|--------|-----------|--------|
| Perfect win rate | >95% with 3+ trades | Flag for review |
| Single-market wallet | 100% of trades in one market | High suspicion |
| Pre-news trading | >50% of trades within 1hr of resolution | Alert |
| Fresh wallet | <7 days old with large positions | Monitor closely |
| Cluster detection | Multiple wallets, coordinated trades | Aggregate as one |

**Note:** ~25% of Polymarket volume may be wash trades (fake volume). Filter wallets with rapid open/close at extreme prices.

---

## Database Schema

### Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  REAL-TIME (Redis)                 PERSISTENT (PostgreSQL)       │
│  ─────────────────                 ───────────────────────       │
│                                                                  │
│  • Deduplication (60-min TTL)      • TimescaleDB hypertables     │
│  • Watchlist tickers (7-day TTL)   • pgvector for embeddings     │
│  • Wallet scores cache             • Signal history              │
│  • Real-time alerts queue          • Wallet performance          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture (Research-Backed)

| Decision | Rationale | Source |
|----------|-----------|--------|
| Redis for dedup | Sub-ms latency, 9.5x faster than pgvector | [Redis Benchmark](https://redis.io/blog/benchmarking-results-for-vector-databases/) |
| SemHash + Model2Vec | 130K samples in 7 sec, semantic matching | [SemHash Blog](https://minishlab.github.io/semhash-blogpost/) |
| TimescaleDB hypertables | 90%+ compression, automatic partitioning | [TimescaleDB Docs](https://docs.timescale.com/use-timescale/latest/compression/) |
| pgvector for historical | Hybrid queries with JOINs, market similarity | [Zilliz Comparison](https://zilliz.com/comparison/pgvector-vs-redis) |

### Core Tables

#### Signals (TimescaleDB Hypertable)

```sql
CREATE TABLE signals (
    time TIMESTAMPTZ NOT NULL,
    flow_id TEXT NOT NULL,           -- 'news', 'sentiment', 'market_intel'
    signal_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    market_id TEXT,
    ticker TEXT,
    PRIMARY KEY (time, flow_id)
);

SELECT create_hypertable('signals', 'time', chunk_time_interval => INTERVAL '1 day');

-- Compression after 7 days
ALTER TABLE signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'flow_id, signal_type'
);
SELECT add_compression_policy('signals', INTERVAL '7 days');
```

#### Raw Messages (with Embeddings)

```sql
CREATE TABLE raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_platform TEXT NOT NULL,    -- 'twitter', 'telegram'
    source_account TEXT NOT NULL,
    external_id TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    embedding vector(256),            -- Model2Vec embedding
    source_timestamp TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    is_duplicate BOOLEAN DEFAULT FALSE,
    UNIQUE (source_platform, external_id)
);

-- HNSW index for similarity search
CREATE INDEX ON raw_messages
    USING hnsw (embedding vector_cosine_ops);
```

#### Markets

```sql
CREATE TABLE markets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,           -- 'polymarket', 'kalshi'
    external_id TEXT NOT NULL,
    condition_id TEXT,                -- Polymarket-specific
    question TEXT NOT NULL,
    description TEXT,
    category TEXT,
    end_date TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    winning_outcome TEXT,             -- 'yes', 'no', or specific outcome
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (platform, external_id)
);
```

#### Wallets & Trades (Flow 3)

```sql
CREATE TABLE wallets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,           -- 'polymarket', 'kalshi'
    address TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_active_at TIMESTAMPTZ NOT NULL,
    is_watched BOOLEAN DEFAULT FALSE,
    UNIQUE (platform, address)
);

CREATE TABLE wallet_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wallet_id UUID REFERENCES wallets(id),
    market_id UUID REFERENCES markets(id),
    direction TEXT NOT NULL,          -- 'yes', 'no'
    price DECIMAL(10,6) NOT NULL,
    size DECIMAL(20,6) NOT NULL,
    traded_at TIMESTAMPTZ NOT NULL,
    -- Filled after resolution
    resolved_at TIMESTAMPTZ,
    pnl DECIMAL(20,6),
    is_win BOOLEAN
);

CREATE TABLE wallet_metrics (
    wallet_id UUID PRIMARY KEY REFERENCES wallets(id),
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    total_pnl DECIMAL(20,6),
    pre_news_trades INTEGER DEFAULT 0,
    pre_news_wins INTEGER DEFAULT 0,
    pre_news_accuracy DECIMAL(5,4),
    insider_score DECIMAL(5,4),       -- 0.0 to 1.0
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for finding profitable wallets
CREATE INDEX idx_wallet_metrics_profitable
    ON wallet_metrics (win_rate, insider_score)
    WHERE total_trades >= 10;
```

### Redis Key Schema

| Key Pattern | Type | TTL | Purpose |
|-------------|------|-----|---------|
| `dedup:{hash}` | String | 60 min | News deduplication |
| `watchlist:tickers` | Set | None | Active ticker watchlist |
| `watchlist:ttl:{ticker}` | String | 7 days | Ticker expiry |
| `wallet:score:{address}` | Hash | 1 hour | Cached insider scores |
| `wallets:watched` | Set | None | Addresses to monitor |
| `alerts:pending` | List | None | Unprocessed trade alerts |

### TimescaleDB Configuration

```sql
-- Retention policy: keep raw signals for 90 days
SELECT add_retention_policy('signals', INTERVAL '90 days');

-- Continuous aggregate for hourly rollups
CREATE MATERIALIZED VIEW signals_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    flow_id,
    signal_type,
    count(*) AS signal_count
FROM signals
GROUP BY bucket, flow_id, signal_type;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('signals_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
```

---

## LLM Configuration

### Multi-Provider Support

The system supports interchangeable LLM providers via PydanticAI's model abstraction:

| Provider | SDK | Base URL | Models | Use Case |
|----------|-----|----------|--------|----------|
| **Claude** | `anthropic` | Default Anthropic API | `claude-sonnet-4-20250514` | Primary - best reasoning |
| **ZAI** | `openai` (compatible) | `https://api.z.ai/api/coding/paas/v4` | `glm-4.7`, `glm-4.7-FlashX`, `glm-4.7-Flash` | Alternative - coding focus |

### ZAI GLM-4.7 Models

| Model | Context | Use Case | Cost |
|-------|---------|----------|------|
| `glm-4.7` | 200K tokens | Flagship, highest performance | Paid |
| `glm-4.7-FlashX` | 200K tokens | Lightweight, high-speed | Affordable |
| `glm-4.7-Flash` | 200K tokens | Lightweight | Free |

ZAI API is OpenAI-compatible. See [GLM-4.7 docs](https://docs.z.ai/guides/llm/glm-4.7).

### Provider Selection

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

def get_llm_model(provider: str = "anthropic"):
    """
    Get LLM model based on provider configuration.

    Environment variables:
    - ANTHROPIC_API_KEY: For Claude models
    - ZAI_API_KEY: For ZAI (OpenAI-compatible)
    - LLM_PROVIDER: "anthropic" or "zai"
    """
    if provider == "anthropic":
        return AnthropicModel(
            model_name=settings.llm_model,  # e.g., "claude-sonnet-4-20250514"
        )
    elif provider == "zai":
        return OpenAIModel(
            model_name=settings.zai_model,  # e.g., "glm-4.7"
            base_url="https://api.z.ai/api/coding/paas/v4",
            api_key=settings.zai_api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Usage with PydanticAI Agent
agent = Agent(
    model=get_llm_model(settings.llm_provider),
    result_type=BreakingClassification,
    system_prompt="You are a financial news analyst...",
)
```

### Configuration

```yaml
# config/llm.yaml
llm:
  provider: "anthropic"  # or "zai"

  anthropic:
    model: "claude-sonnet-4-20250514"
    max_tokens: 4096

  zai:
    base_url: "https://api.z.ai/api/coding/paas/v4"
    model: "glm-4.7"              # or "glm-4.7-FlashX" for speed, "glm-4.7-Flash" for free
    max_tokens: 4096
```

### Environment Variables

```bash
# .env
LLM_PROVIDER=anthropic          # "anthropic" or "zai"

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# ZAI (GLM-4.7, OpenAI-compatible)
ZAI_API_KEY=...
ZAI_BASE_URL=https://api.z.ai/api/coding/paas/v4
ZAI_MODEL=glm-4.7
```

### Fallback Strategy

```python
async def run_with_fallback(agent: Agent, prompt: str):
    """
    Try primary provider, fall back to secondary on failure.
    """
    providers = ["anthropic", "zai"] if settings.llm_provider == "anthropic" else ["zai", "anthropic"]

    for provider in providers:
        try:
            model = get_llm_model(provider)
            result = await agent.run(prompt, model=model)
            return result
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            continue

    raise RuntimeError("All LLM providers failed")
```

---

## Implementation Checklist

### Phase 1: Infrastructure
- [ ] Set up Redis Streams for message queue
- [ ] Set up TimescaleDB for signal storage
- [ ] Configure Telegram bot for notifications
- [ ] Create database migrations (schema above)
- [ ] Set up TimescaleDB hypertables + compression policies
- [ ] Configure pgvector indexes for embeddings
- [ ] Set up Redis key schemas for caching

### Phase 2: Flow 1 (Breaking News)
- [x] Twitter stream integration (TwitterAPI.io)
- [x] Telegram stream integration (Telethon)
- [ ] SemHash deduplication
- [ ] PydanticAI classification prompt
- [ ] Polymarket search integration
- [ ] Kalshi search integration
- [ ] Signal output + Telegram notification

### Phase 3: Flow 2 (Sentiment)
- [ ] Register RSS.app account
- [ ] Create RSS feeds for target subreddits
- [ ] Reddit RSS client
- [ ] Twitter search integration
- [ ] Watchlist management (with TTL)
- [ ] Sentiment classification prompt
- [ ] Aggregation logic
- [ ] 6-hour scheduler
- [ ] Signal output + Telegram notification

### Phase 4: Flow 3 (Prediction Markets)
- [ ] Polymarket scanner (trending, expiring)
- [ ] Kalshi scanner
- [ ] Volume spike detection
- [ ] Wallet discovery backfill job (historical analysis)
- [ ] Wallet metrics calculation pipeline
- [ ] Pre-news trade detection queries
- [ ] Insider score algorithm implementation
- [ ] Real-time WebSocket monitor for watched wallets
- [ ] 15-minute scheduler
- [ ] Signal output + Telegram notification

### Phase 5: Integration
- [ ] Flow 1 → Flow 2 watchlist pipeline
- [ ] Cross-flow signal correlation
- [ ] Backtest framework
- [ ] Performance tracking

---

## Configuration

```yaml
# config/flows.yaml

flow1:
  name: "Breaking News & Analysis"
  sources:
    twitter:
      # News accounts - HIGH urgency
      news:
        - "DeItaone"
        - "FinancialJuice"
      # Analysis accounts - NORMAL urgency
      analysis:
        - "spotgamma"
        - "unusual_whales"
        # ... add more as needed
    telegram:
      # News channels - HIGH urgency
      news:
        - "marketfeed"
        - "disclosetv"
      # Analysis channels - NORMAL urgency
      analysis: []  # Add as needed
  deduplication:
    threshold: 0.85
    window_minutes: 60
  llm:
    profile: "reasoning"  # Use reasoning-optimized model (see LLM Configuration)

flow2:
  name: "Sentiment"
  schedule: "0 */6 * * *"  # Every 6 hours
  sources:
    reddit:
      subreddits: ["wallstreetbets", "stocks", "options"]
    twitter:
      search_enabled: true
  watchlist:
    ttl_days: 7
    max_tickers: 50
  llm:
    profile: "fast"  # Use fast model for bulk classification (see LLM Configuration)

flow3:
  name: "Prediction Markets"
  schedule: "*/15 * * * *"  # Every 15 minutes
  platforms:
    - polymarket
    - kalshi
  thresholds:
    volume_spike_zscore: 2.0
    insider_score_min: 0.5
    expiring_hours: 24
```

---

## API Endpoints (Internal)

```
POST /signals/flow1/emit     # Emit breaking news signal
GET  /signals/flow1/latest   # Get latest signals

POST /signals/flow2/run      # Trigger sentiment analysis
GET  /signals/flow2/latest   # Get latest 6h signal

POST /signals/flow3/run      # Trigger market scan
GET  /signals/flow3/latest   # Get latest 15min signal

GET  /watchlist              # Current watchlist
POST /watchlist/add          # Manual ticker addition
DELETE /watchlist/{ticker}   # Remove ticker
```

---

## Dependencies

```toml
# pyproject.toml additions

[project.dependencies]
# Existing
telethon = "^1.34"
httpx = "^0.27"
pydantic = "^2.6"

# LLM (multi-provider via PydanticAI)
pydantic-ai = "^0.1"           # Agent framework with structured outputs
anthropic = "^0.40"            # Claude provider (Agent SDK)
openai = "^1.50"               # ZAI provider (OpenAI-compatible)

# New for PRD2
feedparser = "^6.0"            # RSS parsing
semhash = "^0.1"               # Deduplication
model2vec = "^0.3"             # Embeddings for semhash
apscheduler = "^3.10"          # Scheduling
aiogram = "^3.4"               # Telegram bot
```

---

## Related Documents

- [[Flow 1 - Breaking News]] - Detailed flow design
- [[Flow 2 - Sentiment]] - Sentiment processing details
- [[Flow 3 - Polymarket Intel]] - Prediction market monitoring
- [[Polymarket Integration]] - API clients
- [[Twitter Accounts]] - Curated account list
- [[Telegram Channels]] - Channel list
- [[Reddit Integration]] - Reddit approach
- [[Deduplication]] - SemHash implementation

---

Back to [[Synesis]]
