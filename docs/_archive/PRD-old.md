# Synesis: Complete Product Requirements Document

**Status:** Active
**Created:** 2026-01-17
**Last Updated:** 2026-01-26
**Version:** 2.1

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Vision & Goals](#2-vision--goals)
3. [System Architecture](#3-system-architecture)
4. [Intelligence Layer](#4-intelligence-layer)
5. [Data Sources](#5-data-sources)
6. [Trading Strategies](#6-trading-strategies)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [API Design](#9-api-design)
10. [Configuration](#10-configuration)
11. [Risk Management](#11-risk-management)
12. [Implementation Plan](#12-implementation-plan)
13. [Appendix A: Polymarket Intelligence](#appendix-a-polymarket-intelligence)
14. [Appendix B: Reddit Integration](#appendix-b-reddit-integration)
15. [Appendix C: Twitter/X Integration](#appendix-c-twitterx-integration)

---

## 1. Executive Summary

Synesis is a real-time financial news analysis and prediction market trading system. It transforms social signals (X/Twitter, Telegram, Reddit) into actionable Polymarket trading decisions using LLM-powered analysis, market matching, and automated execution.

**Core Innovation:** Most tools track WHAT traders do (copy trading, whale alerts). Synesis understands WHY trades happen (signal intelligence) and acts FIRST.

---

## 2. Vision & Goals

### 2.1 Product Vision

Build an autonomous trading system that:

1. **Ingests** real-time financial news from X/Twitter, Telegram, and Reddit
2. **Analyzes** content using LLMs for sentiment, entities, and market impact
3. **Matches** news to relevant Polymarket prediction markets
4. **Detects** insider activity and smart money movements
5. **Executes** reasoned bets based on configurable strategies
6. **Tracks** outcomes and calculates win rates by source/event type

### 2.2 Success Metrics

| Metric                 | Target         | Measurement                              |
| ---------------------- | -------------- | ---------------------------------------- |
| News-to-signal latency | < 5 seconds    | Time from ingestion to signal generation |
| Market match accuracy  | > 80%          | Relevant market found for news events    |
| Signal profitability   | > 55% win rate | Directional accuracy on tracked signals  |
| System uptime          | > 99.5%        | Backend availability                     |

### 2.3 Non-Goals (Phase 1)

- Frontend/dashboard (future phase)
- Multi-exchange arbitrage (Kalshi integration deferred)
- High-frequency trading (< 100ms latency)
- Automated position sizing/Kelly criterion

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SYNESIS BACKEND                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         INGESTION LAYER                               │   │
│  │                                                                       │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │   │  Telegram   │  │  X/Twitter  │  │   Reddit    │                  │   │
│  │   │  Listener   │  │   Stream    │  │   Stream    │                  │   │
│  │   │ (Telethon)  │  │ (httpx/v2)  │  │   (PRAW)    │                  │   │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │   │
│  │          │                │                │                          │   │
│  │          └────────────────┼────────────────┘                          │   │
│  │                           ▼                                           │   │
│  │               ┌───────────────────────┐                               │   │
│  │               │   Unified Message     │                               │   │
│  │               │       Queue           │                               │   │
│  │               │   (Redis Streams)     │                               │   │
│  │               └───────────┬───────────┘                               │   │
│  └───────────────────────────┼───────────────────────────────────────────┘   │
│                              ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       PROCESSING LAYER                                │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐    │   │
│  │   │              CROSS-SOURCE DEDUPLICATOR                       │    │   │
│  │   │                     (SemHash)                                │    │   │
│  │   │                                                              │    │   │
│  │   │  - Same news from Telegram + Twitter = single event          │    │   │
│  │   │  - First source wins, others logged as coverage              │    │   │
│  │   │  - Semantic matching catches paraphrases                     │    │   │
│  │   └──────────────────────┬──────────────────────────────────────┘    │   │
│  │                          │                                           │   │
│  │            ┌─────────────┴─────────────┐                            │   │
│  │            ▼                           ▼                            │   │
│  │   [Duplicate]                    [New Event]                        │   │
│  │   Log source coverage            ┌─────────────┐                    │   │
│  │                                  │ LLM Analyzer│                    │   │
│  │                                  │ - Sentiment │                    │   │
│  │                                  │ - Entities  │                    │   │
│  │                                  │ - Urgency   │                    │   │
│  │                                  └──────┬──────┘                    │   │
│  └─────────────────────────────────────────┼────────────────────────────┘   │
│                                            ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      INTELLIGENCE LAYER                               │   │
│  │                                                                       │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │   │   FLOW 1    │  │   FLOW 2    │  │   FLOW 3    │                  │   │
│  │   │  Breaking   │  │  Sentiment  │  │  Polymarket │                  │   │
│  │   │    News     │  │             │  │    Intel    │                  │   │
│  │   │ SPEED EDGE  │  │ INFO EDGE   │  │INSIDER EDGE │                  │   │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │   │
│  │          └────────────────┼────────────────┘                          │   │
│  │                           ▼                                           │   │
│  │               ┌───────────────────────┐                               │   │
│  │               │    DECISION LAYER     │                               │   │
│  │               │  Signal Synthesizer   │                               │   │
│  │               │ Confidence Calculator │                               │   │
│  │               └───────────┬───────────┘                               │   │
│  └───────────────────────────┼───────────────────────────────────────────┘   │
│                              ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         TRADING LAYER                                 │   │
│  │                                                                       │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │   │   Market    │───▶│  Strategy   │───▶│    Order Executor       │  │   │
│  │   │   Matcher   │    │   Engine    │    │    (CLOB Client)        │  │   │
│  │   │ (Gamma API) │    │             │    │                         │  │   │
│  │   └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐    │
│  │    STORAGE     │  │   STREAMING    │  │        FEEDBACK            │    │
│  │                │  │                │  │                            │    │
│  │  PostgreSQL    │  │  WebSocket     │  │  Outcome Tracker           │    │
│  │  +TimescaleDB  │  │  Price Feed    │  │  Win Rate Analytics        │    │
│  │  +pgvector     │  │                │  │                            │    │
│  │  Redis         │  │                │  │                            │    │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Cross-Source Deduplication

The deduplicator is **unified across Telegram, Twitter, and Reddit**. When the same news event arrives from multiple sources:

1. **First source wins** - Gets full LLM analysis and signal generation
2. **Subsequent sources** - Logged as "coverage" for the same event
3. **Semantic matching** - Catches paraphrases, not just exact text matches

```
Example:
  12:00:01 - Telegram @marketfeed: "FED CUTS RATES BY 25BPS"
  12:00:03 - Twitter @DeItaone: "BREAKING: Federal Reserve cuts interest rates by 0.25%"
  12:00:15 - Reddit r/wallstreetbets: "Holy shit Fed just cut rates"

  Result:
  - Event ID: evt_abc123
  - Primary source: Telegram @marketfeed (processed)
  - Coverage: Twitter @DeItaone, Reddit r/wsb (logged, not re-processed)
```

### 3.3 Component Responsibilities

| Component                     | Responsibility                                | Key Dependencies                      |
| ----------------------------- | --------------------------------------------- | ------------------------------------- |
| **Telegram Listener**         | Real-time message streaming from channels     | Telethon                              |
| **Twitter Stream**            | Filtered tweet streaming from accounts        | httpx (TwitterAPI.io)                 |
| **Reddit Stream**             | Subreddit submission streaming                | PRAW                                  |
| **Unified Queue**             | Merge messages from all sources               | Redis Streams                         |
| **Cross-Source Deduplicator** | Semantic dedup across all platforms           | semhash, model2vec, pgvector          |
| **LLM Analyzer**              | Sentiment, classification, entity extraction  | anthropic, openai                     |
| **Breaking News Analyzer**    | Analyze news for trading relevance            | PydanticAI                            |
| **Impact Mapper**             | Map news to affected instruments              | LLM + Gamma API                       |
| **Mispricing Detector**       | AI vs Market price gap detection              | LLM probability estimation            |
| **Sentiment Aggregator**      | Cross-platform sentiment scoring              | Custom lexicon analyzer, LLM fallback |
| **Divergence Detector**       | Cross-platform sentiment divergences          | Custom algorithms                     |
| **Insider Detector**          | Wallet behavior scoring                       | On-chain data                         |
| **Cluster Analyzer**          | Coordinated wallet detection                  | Louvain clustering                    |
| **Decision Engine**           | Synthesize signals, calculate confidence      | Custom algorithms                     |
| **Market Matcher**            | Find relevant Polymarket markets              | Gamma API                             |
| **Strategy Engine**           | Apply trading strategies to signals           | Custom logic                          |
| **Order Executor**            | Place/cancel orders on CLOB                   | py-clob-client                        |
| **WebSocket Feed**            | Real-time price/orderbook updates             | websockets                            |
| **Outcome Tracker**           | Track market resolutions, calculate win rates | Scheduler, PostgreSQL                 |

---

## 4. Intelligence Layer

The Intelligence Layer is the brain of Synesis - it transforms raw signals into actionable trading decisions through 3 parallel flows.

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYNESIS INTELLIGENCE LAYER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │     FLOW 1      │  │     FLOW 2      │  │     FLOW 3      │            │
│   │  Breaking News  │  │   Sentiment     │  │   Polymarket    │            │
│   │                 │  │                 │  │   Intelligence  │            │
│   │   SPEED EDGE    │  │ INFORMATION EDGE│  │  INSIDER EDGE   │            │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│            │                    │                    │                      │
│            └────────────────────┼────────────────────┘                      │
│                                 ▼                                           │
│                    ┌─────────────────────────┐                              │
│                    │     DECISION LAYER      │                              │
│                    │   Signal Synthesizer    │                              │
│                    │  Confidence Calculator  │                              │
│                    └─────────────────────────┘                              │
│                                 │                                           │
│            ┌────────────────────┼────────────────────┐                      │
│            ▼                    ▼                    ▼                      │
│     TradeSignals            Alerts          CorrelationSignals              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Flow                         | Purpose                                              | Edge Type   | Sources                    |
| ---------------------------- | ---------------------------------------------------- | ----------- | -------------------------- |
| **Flow 1: Breaking News**    | Be first to know, map all affected instruments       | Speed       | Twitter breaking, Telegram |
| **Flow 2: Sentiment**        | Aggregate cross-platform sentiment, find divergences | Information | Reddit, Twitter cashtags   |
| **Flow 3: Polymarket Intel** | Detect insiders, track smart money                   | Insider     | On-chain data (PM/Kalshi)  |

### 4.2 Flow 1: Breaking News Intelligence

**Purpose:** Be FIRST to detect market-moving news → Map ALL affected instruments → Trade before market catches up.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FLOW 1: BREAKING NEWS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOURCES                         ANALYSIS                   OUTPUTS        │
│  ──────────────────────         ────────                   ───────        │
│                                                                             │
│  ┌──────────────┐               ┌─────────────────┐                        │
│  │   TWITTER    │               │  LLM ANALYZER   │       ┌─────────────┐  │
│  │  (streaming) │               │                 │       │  AFFECTED   │  │
│  │              │───┐           │ • Entity        │   ┌──▶│  STOCKS     │  │
│  │ @DeItaone    │   │           │   extraction    │   │   │             │  │
│  │ @Newsquawk   │   │           │ • Event type    │   │   │ AAPL, NVDA  │  │
│  │ @FirstSquawk │   │           │ • Urgency       │   │   └─────────────┘  │
│  └──────────────┘   │           │ • Impact        │   │                    │
│                     ▼           └────────┬────────┘   │   ┌─────────────┐  │
│  ┌──────────────┐  ┌────────┐            │            │   │ CORRELATED  │  │
│  │  TELEGRAM    │  │ DEDUP  │────────────▼────────────┼──▶│   ASSETS    │  │
│  │  (streaming) │─▶│        │   ┌─────────────────┐   │   │             │  │
│  │              │  └────────┘   │  IMPACT MAPPER  │   │   │ TLT, BTC    │  │
│  │ Channels     │               │                 │───┘   └─────────────┘  │
│  └──────────────┘               │ • What stocks?  │                        │
│                                 │ • Correlations? │       ┌─────────────┐  │
│  Target: <5s latency            │ • PM markets?   │──────▶│ POLYMARKET  │  │
│                                 │ • Mispricing?   │       │  MARKETS    │  │
│                                 └─────────────────┘       │             │  │
│                                                           │ +Mispricing │  │
│                                                           └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Implementation Status

| Component                  | Status      | Location                                  |
| -------------------------- | ----------- | ----------------------------------------- |
| Twitter Streaming          | ✅ Done     | `src/synesis/ingestion/twitterapi.py`     |
| Telegram Streaming         | ✅ Done     | `src/synesis/ingestion/telegram.py`       |
| Deduplication              | ❌ To Build | `src/synesis/processing/dedup.py`         |
| LLM Breaking News Analyzer | ❌ To Build | `src/synesis/processing/news_analyzer.py` |
| Impact Mapper              | ❌ To Build | `src/synesis/processing/impact_mapper.py` |
| Mispricing Detector        | ❌ To Build | `src/synesis/processing/mispricing.py`    |

#### Deduplication

Prevent processing the same news twice (critical for multi-source ingestion).

**Strategy:** Content hash + sliding window

```python
class Deduplicator:
    """Prevent duplicate news processing."""

    WINDOW_SECONDS = 3600  # 1 hour dedup window

    async def is_duplicate(self, content: str, source: str) -> bool:
        """Check if content was recently processed."""
        # Normalize: lowercase, remove URLs, collapse whitespace
        normalized = self._normalize(content)
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        key = f"dedup:{content_hash}"

        # Check if exists
        if await self.redis.exists(key):
            return True

        # Mark as seen with TTL
        await self.redis.setex(key, self.WINDOW_SECONDS, source)
        return False

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
        return text
```

**Redis Key Schema:**

| Key Pattern    | Type   | TTL    | Purpose               |
| -------------- | ------ | ------ | --------------------- |
| `dedup:{hash}` | String | 1 hour | Content deduplication |

#### Components

**LLM Breaking News Analyzer:**

```python
class BreakingNewsAnalyzer:
    """Analyze breaking news for trading relevance."""

    async def analyze(self, message: RawMessage) -> NewsAnalysis:
        """
        Input: Raw message from Twitter/Telegram
        Output: Structured analysis
        """
        return NewsAnalysis(
            entities=["Fed", "rate cut", "25bps"],
            event_type="MONETARY_POLICY",
            sentiment="BULLISH",         # For risk assets
            urgency="IMMEDIATE",         # HIGH/MEDIUM/LOW
            impact_magnitude="MAJOR",    # MAJOR/MODERATE/MINOR
            confidence=0.85,
            reasoning="Fed rate cut is bullish for growth stocks...",

            # Ticker Discovery (feeds into Flow 2 watchlist)
            mentioned_tickers=[
                TickerMention(ticker="QQQ", confidence=0.95, context="growth stocks"),
                TickerMention(ticker="TLT", confidence=0.90, context="bond prices"),
            ],
            mentioned_companies=["Federal Reserve"],  # Resolved to tickers downstream
        )

class TickerMention(BaseModel):
    """Ticker extracted from news content (not just $cashtags)."""
    ticker: str
    confidence: float  # How confident LLM is this is a real ticker
    context: str       # "IPO", "earnings", "news", etc.
```

> **Flow 1 → Flow 2 Handoff:** Tickers extracted by the LLM analyzer are automatically added to Flow 2's sentiment watchlist:
>
> - Added to Redis set `watchlist:tickers`
> - 7-day TTL via `watchlist:ttl:{ticker}`
> - Flow 2's Sentiment Aggregator monitors all watchlist tickers
> - Enables tracking emerging tickers (e.g., KIOXIA) without manual config

**Impact Mapper:**

Maps news events to affected instruments using **Polymarket Gamma API search**.

```python
class ImpactMapper:
    """Map news events to affected instruments."""

    GAMMA_API = "https://gamma-api.polymarket.com"

    async def map_impact(self, analysis: NewsAnalysis) -> ImpactMap:
        """
        1. Extract search terms from LLM analysis
        2. Query Polymarket Gamma API for matching markets
        3. Calculate mispricing for each match
        """
        # Step 1: Build search queries from entities
        search_terms = self._build_search_terms(analysis)

        # Step 2: Find matching Polymarket markets
        markets = await self._search_polymarket(search_terms)

        # Step 3: Score relevance and calculate mispricing
        market_matches = []
        for market in markets:
            relevance = await self._score_relevance(market, analysis)
            if relevance > 0.5:
                mispricing = await self._calculate_mispricing(market, analysis)
                market_matches.append(MarketMatch(
                    market_id=market.id,
                    question=market.question,
                    current_price=market.price,
                    relevance_score=relevance,
                    **mispricing
                ))

        return ImpactMap(
            mentioned_tickers=analysis.mentioned_tickers,  # → Flow 2
            polymarket_markets=market_matches,
        )

    async def _search_polymarket(self, terms: list[str]) -> list[Market]:
        """Query Gamma API /search endpoint."""
        markets = []
        async with httpx.AsyncClient() as client:
            for term in terms[:5]:  # Limit queries
                resp = await client.get(
                    f"{self.GAMMA_API}/search",
                    params={"query": term, "limit": 10}
                )
                data = resp.json()
                markets.extend(data.get("markets", []))

        # Dedupe by market_id
        return list({m["id"]: m for m in markets}.values())

    def _build_search_terms(self, analysis: NewsAnalysis) -> list[str]:
        """Extract searchable terms from analysis."""
        terms = []
        # Add entities directly
        terms.extend(analysis.entities)
        # Add event-specific terms
        if analysis.event_type == "MONETARY_POLICY":
            terms.extend(["fed", "rate cut", "interest rate", "fomc"])
        elif analysis.event_type == "ELECTION":
            terms.extend(["election", "president", "vote"])
        # ... more event types
        return terms
```

**Mispricing Detector:**

```python
class MispricingDetector:
    """Detect AI estimate vs market price gaps."""

    async def detect(
        self,
        market: PolymarketMarket,
        news_items: list[NewsAnalysis]
    ) -> MispricingSignal | None:
        """
        Compare LLM probability estimate with current market price.
        Signal when gap > 10%.
        """
        ai_probability = await self.estimate_probability(market, news_items)
        market_price = market.current_price

        gap = ai_probability - market_price

        if abs(gap) < 0.10:
            return None

        return MispricingSignal(
            market_id=market.id,
            market_question=market.question,
            ai_estimate=ai_probability,
            market_price=market_price,
            gap=gap,
            direction="BUY_YES" if gap > 0 else "BUY_NO",
            confidence=self.calculate_confidence(news_items),
            reasoning=f"AI estimates {ai_probability:.0%} based on {len(news_items)} news items"
        )
```

#### Pipeline Orchestration

How components connect end-to-end:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLOW 1 PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INGESTION (Implemented)         PROCESSING (To Build)                       │
│  ───────────────────────         ────────────────────                        │
│                                                                              │
│  TwitterStreamClient ─┐                                                      │
│  .on_tweet(callback)  │          ┌──────────────┐                           │
│                       ├─────────►│ NewsRouter   │                           │
│  TelegramListener ────┘          │              │                           │
│  .on_message(callback)           │ 1. Dedup     │                           │
│                                  │ 2. Classify  │                           │
│                                  │ 3. Route     │                           │
│                                  └──────┬───────┘                           │
│                                         │                                    │
│                          ┌──────────────┼──────────────┐                    │
│                          ▼              ▼              ▼                    │
│                    ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│                    │ BREAKING │  │ ROUTINE  │  │  NOISE   │                │
│                    │  NEWS    │  │  NEWS    │  │  (skip)  │                │
│                    └────┬─────┘  └────┬─────┘  └──────────┘                │
│                         │             │                                     │
│                         ▼             ▼                                     │
│                    ┌─────────────────────────┐                              │
│                    │  BreakingNewsAnalyzer   │                              │
│                    │  (LLM: Claude/GPT)      │                              │
│                    │                         │                              │
│                    │  Extracts:              │                              │
│                    │  • Entities             │                              │
│                    │  • Event type           │                              │
│                    │  • Urgency              │                              │
│                    │  • Tickers ────────────────────► Flow 2 Watchlist     │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│                                ▼                                            │
│                    ┌─────────────────────────┐                              │
│                    │     ImpactMapper        │                              │
│                    │                         │                              │
│                    │  1. Build search terms  │                              │
│                    │  2. Query Gamma API     │◄─── Polymarket /search       │
│                    │  3. Score relevance     │                              │
│                    │  4. Calc mispricing     │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│                                ▼                                            │
│                    ┌─────────────────────────┐                              │
│                    │   MispricingSignal      │───► Trading Layer            │
│                    │   (if gap > 10%)        │                              │
│                    └─────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Wiring Code:**

```python
class Flow1Pipeline:
    """Orchestrates Flow 1: Breaking News Intelligence."""

    def __init__(
        self,
        twitter: TwitterStreamClient,
        telegram: TelegramListener,
        dedup: Deduplicator,
        analyzer: BreakingNewsAnalyzer,
        mapper: ImpactMapper,
        redis: Redis,
    ):
        self.dedup = dedup
        self.analyzer = analyzer
        self.mapper = mapper
        self.redis = redis

        # Wire up callbacks
        twitter.on_tweet(self._process_message)
        telegram.on_message(self._process_message)

    async def _process_message(self, msg: Tweet | TelegramMessage) -> None:
        """Main processing pipeline."""
        text = msg.text
        source = "twitter" if isinstance(msg, Tweet) else "telegram"

        # Step 1: Dedup
        if await self.dedup.is_duplicate(text, source):
            return

        # Step 2: Analyze with LLM
        analysis = await self.analyzer.analyze(msg)

        # Step 3: Skip low-urgency
        if analysis.urgency == "LOW":
            return

        # Step 4: Map to markets
        impact = await self.mapper.map_impact(analysis)

        # Step 5: Feed tickers to Flow 2 watchlist
        for ticker in analysis.mentioned_tickers:
            await self._add_to_watchlist(ticker.ticker)

        # Step 6: Emit trading signals
        for market in impact.polymarket_markets:
            if market.mispricing and abs(market.mispricing) > 0.10:
                await self._emit_signal(market)

    async def _add_to_watchlist(self, ticker: str) -> None:
        """Add ticker to Flow 2 sentiment watchlist."""
        await self.redis.sadd("watchlist:tickers", ticker)
        await self.redis.setex(f"watchlist:ttl:{ticker}", 604800, "1")  # 7 days
```

**Example Flow:**

```
Input:  "@DeItaone: BREAKING - Fed cuts rates 25bps"
         ↓
Dedup:  Check if already processed → NEW
         ↓
Analyze: {entities: ["Fed", "rate cut"], urgency: IMMEDIATE, impact: MAJOR}
         ↓
Map:     Stocks: QQQ↑, TLT↑, Banks↓
         Correlations: BTC↑, GOLD↑
         Polymarket: "Fed cuts Jan?" at 45%, AI says 85%
         ↓
Output:  MispricingSignal(gap=40%, direction=BUY_YES, confidence=0.85)
```

### 4.3 Flow 2: Sentiment Intelligence

**Purpose:** Aggregate sentiment across platforms → Detect shifts and divergences → Find lagging market opportunities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FLOW 2: SENTIMENT INTELLIGENCE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FROM FLOW 1              WATCHLIST                                         │
│  ──────────────           ─────────                                         │
│  ┌──────────────┐        ┌─────────────────────────────────┐               │
│  │NewsAnalysis  │───────▶│      DYNAMIC WATCHLIST          │               │
│  │              │        │                                 │               │
│  │mentioned_    │        │  • Auto-adds tickers from Flow 1│               │
│  │tickers: [...]│        │  • 7-day TTL per ticker         │               │
│  └──────────────┘        │  • Redis set for fast lookup    │               │
│                          └────────────┬────────────────────┘               │
│                                       │                                     │
│                                       ▼                                     │
│  SOURCES                 ┌─────────────────────────────────┐   OUTPUTS     │
│  ───────                 │    For each ticker in watchlist │   ───────     │
│                          └────────────┬────────────────────┘               │
│  ┌──────────────┐                     │                    ┌─────────────┐ │
│  │   REDDIT     │                     │                    │  SENTIMENT  │ │
│  │              │        ┌────────────▼────────────┐       │   SCORES    │ │
│  │ • r/WSB      │───────▶│   SENTIMENT AGGREGATOR  │──────▶│             │ │
│  │ • r/stocks   │        │                         │       │ TSLA: +0.72 │ │
│  │ • r/crypto   │        │  ┌────────┐ ┌────────┐  │       │ NVDA: -0.15 │ │
│  └──────────────┘        │  │ Reddit │ │Twitter │  │       └─────────────┘ │
│                          │  │Sentimnt│ │Sentimnt│  │                       │
│  ┌──────────────┐        │  └───┬────┘ └───┬────┘  │       ┌─────────────┐ │
│  │   TWITTER    │───────▶│      └────┬─────┘       │──────▶│  SENTIMENT  │ │
│  │  (sentiment) │        └───────────┼─────────────┘       │   SHIFTS    │ │
│  │              │                    │                     │             │ │
│  │ • Cashtags   │                    ▼                     │ "TSLA: -0.3 │ │
│  │ • $TSLA etc  │        ┌─────────────────────────┐       │  to +0.7"   │ │
│  └──────────────┘        │  DIVERGENCE DETECTOR    │       └─────────────┘ │
│                          │                         │                       │
│                          │  Reddit ≠ Twitter?      │       ┌─────────────┐ │
│  ┌──────────────┐        │  → Signal opportunity   │──────▶│ DIVERGENCE  │ │
│  │  VELOCITY    │        └─────────────────────────┘       │   ALERTS    │ │
│  │  TRACKER     │                                          │             │ │
│  │              │        ┌─────────────────────────┐       │ "Reddit     │ │
│  │ • Hourly     │───────▶│  SPIKE DETECTOR         │──────▶│  bullish,   │ │
│  │   mentions   │        │                         │       │  Twitter    │ │
│  │ • EMA        │        │  mentions > baseline+3σ │       │  lagging"   │ │
│  │   baseline   │        │  → TrendingSignal       │       └─────────────┘ │
│  └──────────────┘        └─────────────────────────┘                       │
│                                       │                    ┌─────────────┐ │
│                                       └───────────────────▶│  TRENDING   │ │
│                                                            │   ALERTS    │ │
│                                                            │             │ │
│                                                            │ "KIOXIA 8x  │ │
│                                                            │  baseline"  │ │
│                                                            └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Data Flow & Storage

Flow 2 relies on **real-time indexing** of all incoming content by ticker.

**Step 1: Ingest & Index**

Every tweet/post gets indexed when processed:

```python
async def index_by_ticker(content: Tweet | RedditPost) -> None:
    """Index content for sentiment/velocity tracking."""
    # Extract tickers using existing SentimentAnalyzer
    tickers = self.analyzer._extract_tickers(content.text)

    hour_bucket = content.created_at.strftime("%Y%m%d%H")
    source = "twitter" if isinstance(content, Tweet) else "reddit"

    for ticker in tickers:
        # Velocity: Increment hourly counter
        await redis.incr(f"mentions:{ticker}:hourly:{hour_bucket}")
        await redis.expire(f"mentions:{ticker}:hourly:{hour_bucket}", 604800)

        # Sentiment: Store for aggregation
        await redis.lpush(f"{source}:ticker:{ticker}", content.model_dump_json())
        await redis.ltrim(f"{source}:ticker:{ticker}", 0, 999)
```

**Step 2: Query on Demand**

Sentiment Aggregator reads indexed data:

```python
async def get_twitter_sentiment(self, ticker: str, hours: int = 24) -> TickerSentiment:
    # 1. Get stored tweets
    tweets_json = await redis.lrange(f"twitter:ticker:{ticker}", 0, -1)
    tweets = [Tweet.model_validate_json(t) for t in tweets_json]

    # 2. Filter to time window
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    tweets = [t for t in tweets if t.created_at > cutoff]

    # 3. Run through existing SentimentAnalyzer
    scores = [await self.analyzer.analyze(t.text) for t in tweets]

    return TickerSentiment(
        compound=mean(s.compound for s in scores),
        volume=len(tweets),
    )
```

**Redis Key Schema:**

| Key Pattern                             | Type            | TTL    | Purpose           |
| --------------------------------------- | --------------- | ------ | ----------------- |
| `mentions:{ticker}:hourly:{YYYYMMDDHH}` | Counter         | 7 days | Velocity tracking |
| `twitter:ticker:{ticker}`               | List (max 1000) | None   | Tweet storage     |
| `reddit:ticker:{ticker}`                | List (max 1000) | None   | Post storage      |
| `watchlist:tickers`                     | Set             | None   | Active watchlist  |
| `watchlist:ttl:{ticker}`                | String          | 7 days | Ticker expiry     |

**Data Flow Diagram:**

```
INGESTION                           STORAGE                         QUERY
─────────                           ───────                         ─────

Twitter Stream ─┐                   Redis
(40 accounts)   │                   ┌────────────────────────┐
                │   index_by_       │ mentions:TSLA:hourly:* │──► Velocity
                ├──►ticker()───────►│ twitter:ticker:TSLA    │    Tracker
Reddit RSS ─────┘                   │ reddit:ticker:TSLA     │
(rss.app)                           └────────────────────────┘
                                              │
                                              ▼
                                    ┌────────────────────────┐
                                    │  Sentiment Aggregator  │
                                    │                        │
                                    │  1. LRANGE tweets      │
                                    │  2. SentimentAnalyzer  │──► Sentiment
                                    │  3. mean(scores)       │    Scores
                                    └────────────────────────┘
```

#### Components

**Ticker Extractor:**

```python
class TickerExtractor:
    """Extract and validate tickers from text."""

    BLACKLIST = {
        "YOLO", "FOMO", "DD", "HODL", "CEO", "IPO", "ETF",
        "USA", "USD", "THE", "FOR", "AND", "LOL", "WTF",
        "DEFI", "NFT", "DAO", "DEX", "CEX", "KYC", "AML",
        # ... comprehensive blacklist
    }

    def extract(self, text: str) -> list[str]:
        """
        1. Parse cashtags ($TSLA) and bare tickers (TSLA)
        2. Filter blacklist
        3. Validate against real ticker list
        """
        pass
```

**Sentiment Aggregator:**

```python
class SentimentAggregator:
    """Aggregate sentiment across platforms."""

    async def aggregate(self, ticker: str, window_hours: int = 24) -> AggregatedSentiment:
        reddit_sentiment = await self.get_reddit_sentiment(ticker, window_hours)
        twitter_sentiment = await self.get_twitter_sentiment(ticker, window_hours)

        return AggregatedSentiment(
            ticker=ticker,
            reddit=reddit_sentiment,      # -1.0 to +1.0
            twitter=twitter_sentiment,    # -1.0 to +1.0
            combined=self.weighted_average(reddit_sentiment, twitter_sentiment),
            volume={
                "reddit": reddit_sentiment.post_count,
                "twitter": twitter_sentiment.mention_count,
            },
            divergence=self.calculate_divergence(reddit_sentiment, twitter_sentiment),
        )
```

#### Twitter Sentiment Data Sources

Twitter sentiment for watchlist tickers comes from **two sources**:

| Source             | How                                                        | When Used                      | Cost             |
| ------------------ | ---------------------------------------------------------- | ------------------------------ | ---------------- |
| **Stream-derived** | Index tweets from Flow 1's 40 accounts by ticker mentioned | Always (baseline)              | Free             |
| **Cashtag Search** | Poll TwitterAPI.io for `$TICKER` periodically              | High-priority/trending tickers | ~$0.15/1k tweets |

**Implementation:**

```python
class TwitterSentimentFetcher:
    """Fetch Twitter sentiment for tickers via hybrid approach."""

    SEARCH_INTERVAL_MINUTES = 30
    MAX_SEARCH_TICKERS = 20  # Budget: ~$7/day at 100 tweets per ticker

    async def get_twitter_sentiment(self, ticker: str, window_hours: int = 24) -> TickerSentiment:
        # 1. Check stream-derived data first (free)
        stream_data = await self.redis.lrange(f"stream:ticker:{ticker}", 0, -1)

        # 2. If ticker is high-priority, supplement with search
        if await self.is_high_priority(ticker):
            search_data = await self.search_cashtag(ticker, window_hours)
            stream_data.extend(search_data)

        # 3. Run sentiment analysis on combined tweets
        return await self.analyzer.analyze_batch(stream_data)

    async def search_cashtag(self, ticker: str, hours: int = 24) -> list[Tweet]:
        """Search TwitterAPI.io for cashtag mentions."""
        query = f"${ticker} -is:retweet lang:en"
        return await self.twitter_client.advanced_search(
            query=query,
            start_time=datetime.now() - timedelta(hours=hours),
            max_results=100,
        )

    async def is_high_priority(self, ticker: str) -> bool:
        """Determine if ticker warrants search API budget."""
        # High priority if: trending, recently from Flow 1, or manually flagged
        return (
            await self.is_trending(ticker) or
            await self.recently_from_flow1(ticker) or
            ticker in self.priority_watchlist
        )
```

**Stream-derived indexing** (in `ingestion/twitterapi.py`):

```python
# When processing tweets from Flow 1 stream, index by ticker
async def index_tweet_by_tickers(tweet: Tweet, analysis: NewsAnalysis) -> None:
    """Index tweet for sentiment lookup by each mentioned ticker."""
    for ticker in analysis.mentioned_tickers:
        await redis.lpush(f"stream:ticker:{ticker.ticker}", tweet.json())
        await redis.ltrim(f"stream:ticker:{ticker.ticker}", 0, 999)  # Keep last 1000
```

**Cost optimization:**

- Stream-derived data is free (already ingesting for Flow 1)
- Search budget focused on trending/high-priority tickers
- Batch searches: `($TSLA OR $NVDA OR $AAPL)` reduces API calls

**Divergence Detector:**

```python
class DivergenceDetector:
    """Detect cross-platform sentiment divergences."""

    async def detect(self, ticker: str) -> DivergenceSignal | None:
        """
        Signal when platforms disagree significantly.

        Examples:
        - Reddit bullish (+0.7), Twitter neutral (0.0) = Early Reddit signal
        - Twitter bearish (-0.5), Reddit bullish (+0.3) = Conflicting info
        """
        agg = await self.aggregator.aggregate(ticker)

        if abs(agg.divergence) > 0.4:
            return DivergenceSignal(
                ticker=ticker,
                reddit_sentiment=agg.reddit,
                twitter_sentiment=agg.twitter,
                divergence=agg.divergence,
                interpretation=self.interpret(agg),
            )
        return None
```

#### Dynamic Watchlist

The sentiment watchlist is **not static**. Tickers are automatically added from Flow 1 when the LLM Breaking News Analyzer extracts them from incoming news:

```python
class DynamicWatchlist:
    """Auto-expanding ticker watchlist fed by Flow 1."""

    TICKER_TTL_DAYS = 7  # Tickers expire after 7 days of no mentions

    async def add_from_news(self, analysis: NewsAnalysis) -> None:
        """Add tickers from Flow 1 analysis to sentiment watchlist."""
        for ticker_mention in analysis.mentioned_tickers:
            if ticker_mention.confidence > 0.7:
                await self.redis.sadd("watchlist:tickers", ticker_mention.ticker)
                await self.redis.setex(
                    f"watchlist:ttl:{ticker_mention.ticker}",
                    self.TICKER_TTL_DAYS * 86400,  # seconds
                    ticker_mention.context
                )

    async def get_active_tickers(self) -> set[str]:
        """Get all tickers with active TTL."""
        return await self.redis.smembers("watchlist:tickers")

    async def cleanup_expired(self) -> None:
        """Remove tickers whose TTL has expired."""
        for ticker in await self.get_active_tickers():
            if not await self.redis.exists(f"watchlist:ttl:{ticker}"):
                await self.redis.srem("watchlist:tickers", ticker)
```

**Data Flow:**

```
Flow 1 (Breaking News)
    → LLM extracts tickers (not just $cashtags)
    → Auto-add to Flow 2 watchlist (Redis set)
    → Sentiment tracked across Twitter + Reddit
    → 7-day TTL auto-expires stale tickers
```

This enables Synesis to track **emerging tickers** (e.g., KIOXIA IPO) that suddenly appear in news without requiring manual watchlist updates.

#### Velocity Spike Detection

Detect trending tickers by monitoring mention velocity against historical baseline:

```python
class TrendingTickerDetector:
    """Detect trending tickers via mention velocity spikes."""

    # Volume spike detection parameters (research-backed)
    SPIKE_THRESHOLD = 3.0       # K standard deviations (3-4x optimal)
    BASELINE_WINDOW_DAYS = 7    # EMA baseline period
    MIN_MENTIONS_FOR_TREND = 10 # Ignore low-volume tickers

    async def check_ticker_velocity(self, ticker: str) -> TrendingSignal | None:
        """
        Check if ticker is trending based on mention velocity.

        Algorithm: Flag when mentions > baseline + K × std_dev
        - K = 3-4x filters noise while catching real signals
        - Even large-cap stocks spike 8-11x baseline during major events
        """
        baseline = await self.get_baseline(ticker)
        current = await self.get_current_mentions(ticker, hours=1)

        if baseline.count < self.MIN_MENTIONS_FOR_TREND:
            return None

        z_score = (current - baseline.mean) / baseline.std_dev

        if z_score > self.SPIKE_THRESHOLD:
            return TrendingSignal(
                ticker=ticker,
                current_mentions=current,
                baseline_mean=baseline.mean,
                z_score=z_score,
                spike_ratio=current / baseline.mean,
            )
        return None

    async def get_baseline(self, ticker: str) -> TickerBaseline:
        """Get historical baseline using EMA (more responsive than SMA)."""
        mentions = await self.redis.lrange(f"mentions:{ticker}:hourly", 0, -1)
        if not mentions:
            return TickerBaseline(count=0, mean=0, std_dev=1)

        values = [int(m) for m in mentions]
        return TickerBaseline(
            count=len(values),
            mean=self._calculate_ema(values),
            std_dev=self._calculate_std(values),
        )

    async def discover_new_tickers(self) -> list[str]:
        """
        Periodic job: Search for cashtags not in our watchlist.

        Cost: ~$0.15/1k tweets × 100 tweets/30min = $7.20/day
        """
        # Search from high-signal accounts
        query = "(from:KobeissiLetter OR from:unusual_whales OR from:jukan05) $"
        tweets = await self.twitter_client.search(query, limit=100)

        tickers = self._extract_cashtags(tweets)
        watchlist = await self.watchlist.get_active_tickers()

        return [t for t in tickers if t not in watchlist]
```

**Spike Detection Algorithm:**

- Standard approach: flag when `mentions > baseline + K × std_dev`
- **K = 3-4x** is optimal threshold (filters noise, catches real signals)
- Use EMA for baseline (more responsive than SMA to recent changes)
- Even large-cap stocks spike 8-11x baseline during major events

| Spike Ratio | Interpretation    | Action          |
| ----------- | ----------------- | --------------- |
| 3-5x        | Moderate interest | Monitor         |
| 5-8x        | High interest     | Alert           |
| 8x+         | Major event       | Priority signal |

### 4.4 Flow 3: Polymarket Intelligence

**Purpose:** Detect insider activity via on-chain analysis → Track smart money → Copy high-conviction trades.

**Data Sources:** On-chain only - NO Telegram whale alerts. All data from:

- Polymarket contract events (Polygon)
- Kalshi API (when available)
- Direct wallet/position queries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLOW 3: POLYMARKET INTELLIGENCE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOURCES (On-Chain)            DETECTION                    OUTPUTS        │
│  ──────────────────            ─────────                    ───────        │
│                                                                             │
│  ┌──────────────┐             ┌─────────────────┐                          │
│  │  POLYMARKET  │             │                 │         ┌─────────────┐  │
│  │  ON-CHAIN    │             │ INSIDER         │         │  INSIDER    │  │
│  │              │             │ DETECTOR        │         │   ALERTS    │  │
│  │ • Polygon    │──┐          │                 │         │             │  │
│  │   contract   │  │          │ Scoring:        │     ┌──▶│ Score: 85   │  │
│  │   events     │  │          │ +40 Fresh wallet│     │   │ "Fresh      │  │
│  │ • Trade txns │  │          │     + big bet   │     │   │  wallet     │  │
│  │ • Positions  │  │          │ +30 Single mkt  │     │   │  $50k YES"  │  │
│  └──────────────┘  │          │ +25 Perfect     │     │   └─────────────┘  │
│                    │          │     win rate    │     │                    │
│  ┌──────────────┐  │          │ +20 Hidden      │     │   ┌─────────────┐  │
│  │   WALLET     │  │          │     funding     │     │   │   COPY      │  │
│  │   TRACKER    │  │          │                 │─────┼──▶│  SIGNALS    │  │
│  │              │──┼─────────▶│ Thresholds:     │     │   │             │  │
│  │ • Top        │  │          │ 70+ = COPY      │     │   │ "Copy 0x123 │  │
│  │   holders    │  │          │ 50-69 = ALERT   │     │   │  direction" │  │
│  │ • History    │  │          │ 30-49 = MONITOR │     │   └─────────────┘  │
│  │ • Win rates  │  │          └─────────────────┘     │                    │
│  └──────────────┘  │                                  │   ┌─────────────┐  │
│                    │          ┌─────────────────┐     │   │  CLUSTER    │  │
│                    │          │                 │     │   │  ACTIVITY   │  │
│                    └─────────▶│ CLUSTER         │     │   │             │  │
│                               │ ANALYZER        │─────┴──▶│ "5 wallets  │  │
│                               │                 │         │  trading    │  │
│                               │ • Louvain       │         │  together"  │  │
│                               │   clustering    │         └─────────────┘  │
│                               │ • Timing corr   │                          │
│                               │ • Jaccard sim   │                          │
│                               └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Insight: Behavioral Detection

> "You do not spot informed traders by one big bet. You spot them by **behavior**."
> — @k1rallik

Single large bets are obvious and often noise. True insiders reveal themselves through **patterns over time**.

#### Market Filter

**NOT all markets have insider edge.** Filter OUT markets where information asymmetry is unlikely:

```python
class MarketFilter:
    """Filter markets for insider detection relevance."""

    # Markets where insider edge is UNLIKELY (public info, random outcomes)
    EXCLUDED_CATEGORIES = {
        "sports": ["NFL", "NBA", "MLB", "NHL", "Super Bowl", "World Cup", "UFC"],
        "crypto_price": ["BTC to $", "ETH to $", "SOL to $", "price by"],
        "weather": ["temperature", "rainfall", "hurricane category"],
        "entertainment": ["Oscar", "Grammy", "Emmy", "box office"],
    }

    # Markets where insider edge IS LIKELY (private info, human decisions)
    INCLUDED_CATEGORIES = {
        "politics": ["election", "president", "congress", "legislation", "resign"],
        "corporate": ["CEO", "merger", "acquisition", "layoff", "bankruptcy"],
        "geopolitics": ["war", "ceasefire", "sanctions", "treaty", "invasion"],
        "regulatory": ["FDA approval", "SEC", "indictment", "verdict", "ruling"],
    }

    def should_monitor(self, market: PolymarketMarket) -> bool:
        """Returns True if market has potential for insider information."""
        question = market.question.lower()

        # Exclude sports, crypto price, etc.
        for category, keywords in self.EXCLUDED_CATEGORIES.items():
            if any(kw.lower() in question for kw in keywords):
                return False

        # Include politics, corporate, etc.
        for category, keywords in self.INCLUDED_CATEGORIES.items():
            if any(kw.lower() in question for kw in keywords):
                return True

        return False  # Default: don't monitor unknown categories
```

#### Components

**Insider Detector:**

```python
class InsiderDetector:
    """Score wallets for insider probability based on BEHAVIOR, not single bets."""

    # === TRADITIONAL SIGNALS (single-trade indicators) ===
    SCORING = {
        "fresh_wallet_large_bet": 40,   # Account <24h + bet >$10k
        "single_market_focus": 30,       # Only trades ONE market
        "perfect_win_rate_niche": 25,    # 100% in specific category
        "hidden_funding_source": 20,     # No traceable deposit history
        "wash_trading_cover": 15,        # Buy/sell same market
    }

    # === BEHAVIORAL SIGNALS (patterns over time) ===
    # Per @k1rallik: "You spot them by behavior"
    BEHAVIORAL_SCORING = {
        "warmup_trades": 25,             # Small test bets before big position
        "silent_accumulation": 30,       # Gradual position building over hours/days
        "contrarian_vs_consensus": 35,   # Betting against 80%+ market skew
    }

    # === PROBABILITY-AWARE MULTIPLIERS ===
    # Betting against strong consensus is MORE suspicious
    CONSENSUS_MULTIPLIERS = {
        "90%+": 1.5,   # $10k against 90% consensus = 1.5x suspicion
        "80-89%": 1.3,
        "70-79%": 1.1,
        "60-69%": 1.0,  # Baseline
        "<60%": 0.8,    # Less suspicious to bet against weak consensus
    }

    def __init__(self):
        self.wallet_memory: dict[str, WalletBehaviorState] = {}  # Stateful tracking

    async def score_wallet(self, wallet: str, market_id: str) -> InsiderScore:
        history = await self.fetcher.get_wallet_history(wallet)
        position = await self.fetcher.get_position(wallet, market_id)
        market = await self.fetcher.get_market(market_id)

        score = 0
        flags = []

        # --- Traditional single-trade signals ---
        if history.account_age_hours < 24 and position.size > 10_000:
            score += self.SCORING["fresh_wallet_large_bet"]
            flags.append("FRESH_WALLET_LARGE_BET")

        if history.markets_traded == 1:
            score += self.SCORING["single_market_focus"]
            flags.append("SINGLE_MARKET_FOCUS")

        # --- Behavioral signals (patterns over time) ---
        behavior = await self._get_wallet_behavior(wallet, market_id)

        # Warmup trades: Small bets before big position
        if behavior.has_warmup_pattern:
            score += self.BEHAVIORAL_SCORING["warmup_trades"]
            flags.append("WARMUP_TRADES")

        # Silent accumulation: Gradual position building
        if behavior.has_accumulation_pattern:
            score += self.BEHAVIORAL_SCORING["silent_accumulation"]
            flags.append("SILENT_ACCUMULATION")

        # Contrarian vs consensus: Betting against market skew
        consensus = market.yes_price if position.side == "NO" else (1 - market.yes_price)
        if consensus > 0.80 and position.size > 5_000:
            score += self.BEHAVIORAL_SCORING["contrarian_vs_consensus"]
            flags.append("CONTRARIAN_VS_CONSENSUS")

            # Apply probability-aware multiplier
            multiplier = self._get_consensus_multiplier(consensus)
            score = int(score * multiplier)

        # Update stateful memory
        self._update_wallet_memory(wallet, position, behavior)

        return InsiderScore(
            wallet=wallet,
            score=score,
            flags=flags,
            action="COPY" if score >= 70 else "ALERT" if score >= 50 else "MONITOR"
        )

    async def _get_wallet_behavior(self, wallet: str, market_id: str) -> WalletBehavior:
        """Analyze wallet behavior patterns over time."""
        trades = await self.fetcher.get_wallet_trades(wallet, market_id)

        # Warmup pattern: 2+ small trades (<$500) before large trade (>$5k)
        has_warmup = self._detect_warmup_pattern(trades)

        # Accumulation pattern: 3+ trades over 24h+ with increasing size
        has_accumulation = self._detect_accumulation_pattern(trades)

        return WalletBehavior(
            has_warmup_pattern=has_warmup,
            has_accumulation_pattern=has_accumulation,
            trade_count=len(trades),
            time_span_hours=self._calculate_time_span(trades),
        )

    def _detect_warmup_pattern(self, trades: list[Trade]) -> bool:
        """Detect small test bets before large position."""
        if len(trades) < 3:
            return False

        # Sort by timestamp
        trades = sorted(trades, key=lambda t: t.timestamp)

        # Check for 2+ small trades followed by large trade
        small_trades = [t for t in trades[:-1] if t.size < 500]
        final_trade = trades[-1]

        return len(small_trades) >= 2 and final_trade.size > 5_000

    def _detect_accumulation_pattern(self, trades: list[Trade]) -> bool:
        """Detect gradual position building over time."""
        if len(trades) < 3:
            return False

        trades = sorted(trades, key=lambda t: t.timestamp)
        time_span = (trades[-1].timestamp - trades[0].timestamp).total_seconds() / 3600

        # Must span at least 24 hours
        if time_span < 24:
            return False

        # Check for increasing position sizes
        sizes = [t.size for t in trades]
        return all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1))

    def _get_consensus_multiplier(self, consensus: float) -> float:
        """Get suspicion multiplier based on market consensus."""
        if consensus >= 0.90:
            return self.CONSENSUS_MULTIPLIERS["90%+"]
        elif consensus >= 0.80:
            return self.CONSENSUS_MULTIPLIERS["80-89%"]
        elif consensus >= 0.70:
            return self.CONSENSUS_MULTIPLIERS["70-79%"]
        elif consensus >= 0.60:
            return self.CONSENSUS_MULTIPLIERS["60-69%"]
        else:
            return self.CONSENSUS_MULTIPLIERS["<60%"]

    def _update_wallet_memory(self, wallet: str, position, behavior) -> None:
        """Update stateful memory for long-term tracking."""
        if wallet not in self.wallet_memory:
            self.wallet_memory[wallet] = WalletBehaviorState()

        self.wallet_memory[wallet].update(position, behavior)
```

**Cluster Analyzer:**

```python
class ClusterAnalyzer:
    """Detect coordinated wallet activity."""

    async def find_clusters(self, market_id: str) -> list[WalletCluster]:
        """Use graph analysis to find wallets trading together."""
        top_holders = await self.get_top_holders(market_id)

        edges = []
        for wallet_a, wallet_b in combinations(top_holders, 2):
            timing_corr = self.timing_correlation(wallet_a, wallet_b)
            position_sim = self.jaccard_similarity(
                self.get_positions(wallet_a),
                self.get_positions(wallet_b)
            )

            if timing_corr > 0.8 or position_sim > 0.7:
                edges.append((wallet_a, wallet_b))

        clusters = self.louvain_communities(edges)

        return [
            WalletCluster(wallets=c, likely_single_entity=True)
            for c in clusters if len(c) >= 3
        ]
```

### 4.5 Decision Layer

**Purpose:** Synthesize outputs from all 3 flows → Calculate confidence → Generate actionable outputs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DECISION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│      FLOW 1                  FLOW 2                  FLOW 3                 │
│   Breaking News            Sentiment             Polymarket Intel           │
│        │                       │                       │                    │
│        ▼                       ▼                       ▼                    │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    SIGNAL SYNTHESIZER                             │     │
│   │                                                                   │     │
│   │  Combines signals when multiple flows agree:                      │     │
│   │                                                                   │     │
│   │  Example: Breaking news (Fed cut) + Sentiment (bullish growth)   │     │
│   │           + Insider (buying YES on Fed market)                   │     │
│   │           = HIGH CONFIDENCE multi-flow signal                    │     │
│   └───────────────────────────┬──────────────────────────────────────┘     │
│                               │                                             │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                  CONFIDENCE CALCULATOR                            │     │
│   │                                                                   │     │
│   │   source_reliability      (0.00 - 0.25)  ─┐                      │     │
│   │   multi_flow_agreement    (0.00 - 0.30)  ─┼── final_confidence   │     │
│   │   signal_strength         (0.00 - 0.25)  ─┤      (0.0 - 1.0)     │     │
│   │   market_conditions       (0.00 - 0.20)  ─┘                      │     │
│   │                                                                   │     │
│   │   BONUS: +0.15 when 2+ flows agree on same direction             │     │
│   │   BONUS: +0.10 when insider + breaking news align                │     │
│   └───────────────────────────┬──────────────────────────────────────┘     │
│                               │                                             │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    OUTPUT GENERATOR                               │     │
│   │                                                                   │     │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │     │
│   │   │   TRADE     │   │    ALERT    │   │ CORRELATION │            │     │
│   │   │  SIGNALS    │   │             │   │   SIGNALS   │            │     │
│   │   │             │   │ • INSIDER   │   │             │            │     │
│   │   │ Actionable  │   │ • SENTIMENT │   │ Polymarket  │            │     │
│   │   │ with size,  │   │ • BREAKING  │   │ → Stocks    │            │     │
│   │   │ direction,  │   │ • MISPRICING│   │ direction   │            │     │
│   │   │ reasoning   │   │ • CLUSTER   │   │ guidance    │            │     │
│   │   └─────────────┘   └─────────────┘   └─────────────┘            │     │
│   └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Output Types

**TradeSignal (Actionable):**

```python
@dataclass
class TradeSignal:
    id: uuid
    timestamp: datetime

    # What to trade
    venue: Literal["polymarket", "kalshi", "stock", "crypto"]
    instrument: str                    # market_id, ticker, token
    direction: Literal["LONG", "SHORT", "BUY_YES", "BUY_NO"]

    # Signal strength
    confidence: float                  # 0.0 - 1.0
    edge_estimate: float               # Expected value per dollar
    urgency: Literal["IMMEDIATE", "HOURS", "DAYS"]

    # Position guidance
    suggested_size: float              # As % of bankroll
    max_size: float
    entry_price_target: float | None

    # Attribution
    source_flows: list[str]            # ["breaking_news", "insider"]
    reasoning: str                     # LLM-generated explanation
```

**Alert (Informational):**

```python
@dataclass
class Alert:
    type: Literal[
        "INSIDER_DETECTED",
        "SENTIMENT_SHIFT",
        "BREAKING_NEWS",
        "MARKET_MISPRICING",
        "CLUSTER_ACTIVITY",
        "DIVERGENCE_DETECTED",
    ]
    severity: Literal["INFO", "WARNING", "CRITICAL"]

    affected_instruments: list[str]
    evidence: list[str]
    suggested_actions: list[str]
```

**CorrelationSignal:**

```python
@dataclass
class CorrelationSignal:
    # Polymarket as oracle
    polymarket_market: str
    polymarket_price: float
    polymarket_direction: str

    # Traditional asset implications
    correlated_assets: list[AssetCorrelation]
    # e.g., "Fed cuts" → Long TLT, Long QQQ
    # e.g., "Trump wins" → Long DJT, crypto
```

#### Confidence Calculator

```python
class ConfidenceCalculator:
    """Calculate final confidence from multiple signals."""

    WEIGHTS = {
        "source_reliability": 0.25,
        "multi_flow_agreement": 0.30,
        "signal_strength": 0.25,
        "market_conditions": 0.20,
    }

    BONUSES = {
        "two_plus_flows_agree": 0.15,
        "insider_plus_breaking": 0.10,
        "all_three_flows": 0.20,
    }

    def calculate(self, signals: list[FlowSignal]) -> float:
        base_score = self._weighted_base(signals)
        bonuses = self._calculate_bonuses(signals)
        return min(base_score + bonuses, 1.0)
```

### 4.6 Unified Ticker Discovery + Tracking

The system uses a **three-pronged approach** to discover and track tickers:

| Approach                     | How                                                                 | Pros                                         | Cons                                        |
| ---------------------------- | ------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| **LLM Entity Extraction**    | Extract companies/tickers from existing stream (not just $cashtags) | Zero additional API cost, uses existing data | Only catches tickers from followed accounts |
| **Periodic Cashtag Search**  | Poll TwitterAPI.io for trending cashtags every 30 min               | Catches tickers outside follow list          | Additional API cost (~$7.20/day)            |
| **Signal Account Expansion** | Add accounts that surface trending tickers                          | Low effort, high signal                      | Still limited to those accounts' coverage   |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED TICKER DISCOVERY + TRACKING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Twitter Stream (40+ accounts)                                              │
│  @KobeissiLetter, @DeItaone, @unusual_whales, etc.                         │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       LLM ANALYZER                                   │   │
│  │                                                                      │   │
│  │  Input: Raw tweet                                                    │   │
│  │  Output:                                                             │   │
│  │   - NewsAnalysis (sentiment, urgency, impact)                        │   │
│  │   - Extracted tickers: [$KIOXIA, $NVDA, ...]                        │   │
│  │   - Company names → resolved to tickers                              │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│            ┌────────────────┼────────────────┐                              │
│            ▼                ▼                ▼                              │
│     ┌──────────┐     ┌──────────┐     ┌──────────┐                         │
│     │  FLOW 1  │     │  FLOW 2  │     │ VELOCITY │                         │
│     │  SPEED   │     │ SENTIMENT│     │ TRACKER  │                         │
│     │  EDGE    │     │ WATCHLIST│     │ (Redis)  │                         │
│     │          │     │          │     │          │                         │
│     │ Immediate│     │ Auto-add │     │ Track    │                         │
│     │ trading  │     │ tickers  │     │ mention  │                         │
│     │ decision │     │ for x-   │     │ velocity │                         │
│     │          │     │ platform │     │ per hour │                         │
│     │          │     │ sentiment│     │          │                         │
│     └────┬─────┘     └────┬─────┘     └────┬─────┘                         │
│          │                │                │                                │
│          │                ▼                ▼                                │
│          │         ┌─────────────────────────────┐                         │
│          │         │    SENTIMENT AGGREGATOR     │                         │
│          │         │                             │                         │
│          │         │  For each ticker in         │                         │
│          │         │  watchlist:                 │                         │
│          │         │  - Twitter sentiment        │                         │
│          │         │  - Reddit sentiment         │                         │
│          │         │  - Mention velocity         │                         │
│          │         │  - Divergence detection     │                         │
│          │         └──────────────┬──────────────┘                         │
│          │                        │                                        │
│          ▼                        ▼                                        │
│     TradeSignal           SentimentSignal                                  │
│     (immediate)           (confirmation)                                   │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Tickers discovered in Flow 1 automatically feed into Flow 2's watchlist, creating a self-expanding system that tracks emerging tickers without manual configuration.

#### Cashtag Collision Handling

Same ticker can mean different things (e.g., $COIN = Coinbase or generic "coin"):

```python
class TickerResolver:
    """Resolve ambiguous tickers using LLM classification + context."""

    COLLISION_PATTERNS = {
        "COIN": ["coinbase", "cryptocurrency"],  # Context keywords
        "AI": ["artificial intelligence", "c3.ai"],
        "META": ["facebook", "metaverse"],
    }

    async def resolve(self, ticker: str, context: str) -> ResolvedTicker:
        """Use LLM to disambiguate ticker based on surrounding context."""
        if ticker in self.COLLISION_PATTERNS:
            # LLM classification based on context
            resolution = await self.llm.classify(
                ticker=ticker,
                context=context,
                possible_meanings=self.COLLISION_PATTERNS[ticker]
            )
            return resolution
        return ResolvedTicker(ticker=ticker, confidence=1.0)
```

---

## 5. Data Sources

### 5.1 Twitter/X

**Role:** Primary breaking news source with fastest signal latency.

#### Key Accounts

| Category               | Accounts                                        | Focus                        |
| ---------------------- | ----------------------------------------------- | ---------------------------- |
| **Financial News**     | @DeItaone, @Newsquawk, @FirstSquawk, @zerohedge | Breaking market news         |
| **Crypto**             | @WatcherGuru, @whale_alert, @lookonchain        | Crypto news + whale tracking |
| **Prediction Markets** | @PolyAlertHub, @unusual_whales, @DonOfDAOs      | PM whale alerts              |
| **Official**           | @WhiteHouse, @federalreserve, @SECGov           | Government announcements     |

#### API Configuration

Using TwitterAPI.io ($0.15/1k tweets):

```python
TWITTER_API_KEY=your_twitterapi_io_key
TWITTER_API_BASE_URL=https://api.twitterapi.io
TWITTER_ACCOUNTS=DeItaone,Newsquawk,FirstSquawk,zerohedge
```

#### Signal Characteristics

| Aspect          | Value                              |
| --------------- | ---------------------------------- |
| **Latency**     | Fastest (seconds)                  |
| **Signal Type** | News, announcements, insider hints |
| **Best For**    | Event detection, primary sources   |
| **Noise Level** | High (bots, spam, manipulation)    |

### 5.2 Telegram

**Role:** Fast news delivery, often ahead of or parallel to Twitter.

#### Key Channels

| Channel         | Focus                 |
| --------------- | --------------------- |
| @marketfeed     | General market news   |
| @disclosetv     | Breaking geopolitical |
| @newpolymarkets | New PM market alerts  |
| @YNSignals      | PM alpha signals      |
| t.me/PoIytrage  | Arbitrage alerts      |

#### Configuration

```python
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION_NAME=synesis
TELEGRAM_CHANNELS=marketfeed,disclosetv
```

### 5.3 Reddit (via RSS)

**Role:** Sentiment depth and community consensus. Complements Twitter's speed with discussion context.

**Why RSS instead of Reddit API:**

- Reddit API requires OAuth credentials and has strict rate limits (60 req/min)
- Native Reddit RSS feeds (`.rss` suffix) return 429 errors due to aggressive rate limiting
- [rss.app](https://rss.app) handles rate limit complexity and provides reliable, clean feeds

#### What rss.app Does

rss.app is a service that generates reliable RSS feeds from websites that either don't have good RSS support or rate-limit heavily. Instead of hitting Reddit's API directly:

1. **You create feeds** on rss.app by pasting subreddit URLs
2. **rss.app scrapes Reddit** on their infrastructure (handles rate limits)
3. **You poll rss.app feeds** at regular intervals (clean, reliable data)

#### Key Advantages Over Twitter

- Threaded discussions provide context depth
- Upvote/downvote signals indicate community consensus
- Subreddit-specific targeting reduces noise
- Comment trees reveal sentiment evolution
- **~12 hours slower than Twitter for viral content** - use for sentiment confirmation, not breaking news

#### Target Subreddits

| Category       | Subreddits                                         | rss.app Feed Type |
| -------------- | -------------------------------------------------- | ----------------- |
| **Financial**  | r/wallstreetbets, r/stocks, r/options, r/investing | Hot + New         |
| **Crypto**     | r/CryptoCurrency, r/Bitcoin, r/ethereum            | Hot + New         |
| **Prediction** | r/polymarket, r/predictit                          | New only          |

#### rss.app Configuration

**Recommended Plan:** Developer ($16.64/mo)

- 100 feeds (enough for 10+ subreddits × multiple sorts)
- 15-minute refresh rate
- API access (1000 ops/month, 1 req/sec)
- Alerts to Discord/Telegram/Slack

**Feed Setup:**

1. Create feeds for each subreddit: `https://reddit.com/r/wallstreetbets/new`
2. rss.app generates hosted feed URL: `https://rss.app/feeds/xxxxx.xml`
3. Poll these feeds from Synesis every 5 minutes (faster than rss.app refresh)

#### Environment Variables

```bash
# rss.app (Reddit RSS feeds)
RSS_APP_API_KEY=your_rss_app_api_key
RSS_APP_FEEDS=feed_id_1,feed_id_2,feed_id_3

# Alternative: Direct feed URLs (no API needed)
REDDIT_RSS_FEEDS=https://rss.app/feeds/abc.xml,https://rss.app/feeds/def.xml

# Polling configuration
REDDIT_POLL_INTERVAL_SECONDS=300  # 5 minutes (rss.app refreshes every 15min)
```

#### RSS Parsing Challenges

RSS feeds are notoriously messy. Key issues to handle:

| Issue                  | Solution                                                   |
| ---------------------- | ---------------------------------------------------------- |
| **Malformed XML**      | Use `fastfeedparser` with lxml (10x faster, more tolerant) |
| **Missing fields**     | Defensive parsing with fallbacks                           |
| **Inconsistent dates** | `python-dateutil` for flexible parsing                     |
| **HTML in content**    | `BeautifulSoup` for text extraction                        |
| **Encoding issues**    | Auto-detection via feedparser/fastfeedparser               |
| **Duplicate entries**  | Dedupe by `guid` or `link` hash                            |

#### Latency Expectations

| Component                    | Latency     |
| ---------------------------- | ----------- |
| Reddit post created          | t=0         |
| rss.app detects (worst case) | t+15min     |
| Synesis polls rss.app        | t+15-20min  |
| **Total worst case**         | ~20 minutes |

This is fine for **sentiment analysis** (not breaking news). Twitter handles speed; Reddit handles depth.

### 5.4 Cross-Platform Signal Strategy

**Three-Layer Signal Validation:**

1. **Layer 1: News Detection (Twitter)** - Breaking news, cashtag spikes
2. **Layer 2: Sentiment Confirmation (Reddit)** - Community reaction, upvote velocity
3. **Layer 3: Market Validation (Polymarket)** - Whale positions, odds movement

**Signal Strength Determination:**

| Twitter Volume | Reddit Volume | Signal Strength | Reason                                 |
| -------------- | ------------- | --------------- | -------------------------------------- |
| High           | Zero          | HIGH            | Twitter-first, Reddit hasn't caught up |
| High           | Low (<5)      | MEDIUM          | Early cross-platform correlation       |
| High           | High          | LOW             | Already widely known                   |

---

## 6. Trading Strategies

### 6.1 Strategy Overview

| Strategy                   | Type        | Risk   | Expected Edge                        |
| -------------------------- | ----------- | ------ | ------------------------------------ |
| **News-Driven Sentiment**  | Directional | Medium | LLM analysis before market prices in |
| **Sum-to-One Arbitrage**   | Risk-Free   | None   | YES + NO < $1.00                     |
| **Multi-Outcome Dutching** | Risk-Free   | None   | All outcomes sum < $1.00             |
| **Insider Copy-Trading**   | Directional | Medium | Follow high-score insider wallets    |
| **Market Making**          | Neutral     | Medium | Bid-ask spread capture               |

### 6.2 News-Driven Sentiment Strategy

```python
class NewsDrivenStrategy(BaseStrategy):
    """Trades Polymarket based on LLM analysis of breaking news."""

    async def evaluate(self, signal: Signal) -> Optional[Trade]:
        markets = await self.matcher.find_markets(
            keywords=signal.entities.keywords,
            categories=signal.classification.event_categories
        )

        if not markets:
            return None

        for market in markets:
            current_price = await self.clob.get_midpoint(market.yes_token_id)

            if signal.classification.sentiment == "bullish":
                target_prob = min(current_price + 0.10, 0.95)
                side = "YES"
            else:
                target_prob = max(current_price - 0.10, 0.05)
                side = "NO"

            edge = abs(target_prob - current_price)

            if edge > 0.05 and signal.confidence > 0.7:
                return Trade(
                    market_id=market.condition_id,
                    side=side,
                    size=self.calculate_size(edge, signal.confidence),
                    rationale=f"News-driven: {signal.summary}"
                )

        return None
```

### 6.3 Sum-to-One Arbitrage

```python
class ArbitrageStrategy(BaseStrategy):
    """Detects risk-free arbitrage when YES + NO < $1.00."""

    async def scan_markets(self) -> List[ArbitrageOpportunity]:
        markets = await self.gamma.get_active_markets(
            liquidity_num_min=10000,
            active=True
        )

        opportunities = []
        for market in markets:
            orderbook = await self.clob.get_orderbook(market.condition_id)

            yes_ask = orderbook.yes.ask
            no_ask = orderbook.no.ask
            total_cost = yes_ask + no_ask

            if total_cost < 0.98:  # > 2% profit after fees
                profit_pct = (1.0 - total_cost) * 100
                opportunities.append(ArbitrageOpportunity(
                    market=market,
                    yes_price=yes_ask,
                    no_price=no_ask,
                    profit_pct=profit_pct,
                    max_size=min(orderbook.yes.ask_size, orderbook.no.ask_size)
                ))

        return sorted(opportunities, key=lambda x: -x.profit_pct)
```

### 6.4 AI vs Market Mispricing

When AI assessment differs significantly from market price, that's tradeable alpha:

```python
def calculate_mispricing_signal(market_id: str, news_items: list) -> MispricingSignal | None:
    ai_probability = aggregate_news_sentiment(news_items)
    market_price = get_current_price(market_id)

    discrepancy = ai_probability - market_price

    if abs(discrepancy) < 0.10:
        return None

    source_diversity = len(set(n.source for n in news_items))
    recency_weight = calculate_recency_weight(news_items)

    confidence = min(
        (len(news_items) / 10) *
        (source_diversity / 5) *
        recency_weight,
        1.0
    )

    return MispricingSignal(
        market_id=market_id,
        ai_estimate=ai_probability,
        market_price=market_price,
        discrepancy=discrepancy,
        direction="BUY_YES" if discrepancy > 0 else "BUY_NO",
        edge=abs(discrepancy),
        confidence=confidence,
    )
```

**Trading Logic:**

| Discrepancy | Confidence       | Action                        |
| ----------- | ---------------- | ----------------------------- |
| >15% gap    | High (>0.7)      | Auto-execute (if enabled)     |
| 10-15% gap  | High (>0.7)      | Strong alert, recommend trade |
| 10-15% gap  | Medium (0.5-0.7) | Alert, manual review          |
| <10% gap    | Any              | Log only, no action           |

---

## 7. Tech Stack

### 7.1 Core Technologies

| Category            | Technology     | Version | Rationale                |
| ------------------- | -------------- | ------- | ------------------------ |
| **Language**        | Python         | 3.12+   | Async support, ecosystem |
| **Package Manager** | uv             | latest  | 10-100x faster than pip  |
| **Web Framework**   | FastAPI        | 0.115+  | Async-first, type hints  |
| **Task Queue**      | Celery + Redis | 5.4+    | Distributed tasks        |
| **Message Queue**   | Redis Streams  | 7+      | Lightweight pub/sub      |

### 7.2 Data & Storage

| Category        | Technology                             | Purpose                                     |
| --------------- | -------------------------------------- | ------------------------------------------- |
| **Database**    | PostgreSQL 16 + TimescaleDB + pgvector | All data (relational, time-series, vectors) |
| **DB Driver**   | asyncpg (raw SQL)                      | Direct queries for speed, no ORM overhead   |
| **Cache/Queue** | Redis                                  | Message queue, rate limits, cache           |
| **Migrations**  | SQL files in `database/`               | Simple, version-controlled schema changes   |

### 7.3 External APIs

| Service        | SDK/Library           | Purpose                              |
| -------------- | --------------------- | ------------------------------------ |
| **Polymarket** | py-clob-client        | Trading, orderbook                   |
| **Polymarket** | Gamma API (REST)      | Market discovery                     |
| **Polymarket** | WebSocket             | Real-time prices                     |
| **Telegram**   | Telethon              | Channel streaming                    |
| **X/Twitter**  | TwitterAPI.io (httpx) | Tweet streaming ($0.15/1k tweets)    |
| **Reddit**     | PRAW                  | Subreddit streaming                  |
| **LLM**        | PydanticAI            | Structured analysis (Claude, OpenAI) |
| **Embeddings** | PydanticAI / pgvector | Vector generation & similarity       |

### 7.4 Infrastructure

| Category             | Technology           | Purpose                  |
| -------------------- | -------------------- | ------------------------ |
| **Containerization** | Docker               | Development & deployment |
| **Orchestration**    | Docker Compose       | Local multi-service      |
| **Monitoring**       | Prometheus + Grafana | Metrics                  |
| **Logging**          | structlog            | Structured JSON logs     |
| **Secrets**          | python-dotenv        | Environment variables    |

---

## 8. Project Structure

```
synesis/
├── pyproject.toml              # Project config & dependencies
├── uv.lock                     # Locked dependencies
├── .env.example                # Environment template
├── docker-compose.yml          # Local development stack
├── Dockerfile                  # Backend container
│
├── src/
│   └── synesis/
│       ├── __init__.py
│       ├── main.py             # FastAPI application entry
│       ├── config.py           # Settings via pydantic-settings
│       │
│       ├── core/               # Shared core utilities
│       │   ├── __init__.py
│       │   ├── events.py       # Event bus (Redis Streams)
│       │   ├── logging.py      # structlog configuration
│       │   ├── exceptions.py   # Custom exceptions
│       │   └── dependencies.py # FastAPI dependencies
│       │
│       ├── ingestion/          # Data ingestion layer
│       │   ├── __init__.py
│       │   ├── telegram.py     # Telethon listener
│       │   ├── twitterapi.py   # X/Twitter stream (TwitterAPI.io)
│       │   ├── reddit.py       # Reddit RSS poller (via rss.app)
│       │   ├── webhooks.py     # Webhook receivers
│       │   └── router.py       # Ingestion API routes
│       │
│       ├── processing/         # Analysis & enrichment
│       │   ├── __init__.py
│       │   ├── deduplication.py    # Cross-source semantic dedup
│       │   ├── classifier.py       # LLM classification
│       │   ├── entities.py         # Entity extraction
│       │   ├── embeddings.py       # Vector generation
│       │   └── pipeline.py         # Orchestrates processing
│       │
│       ├── intelligence/       # Intelligence Layer (3 Flows)
│       │   ├── __init__.py
│       │   │
│       │   ├── breaking_news/  # FLOW 1: Breaking News
│       │   │   ├── __init__.py
│       │   │   ├── analyzer.py     # LLM news analysis
│       │   │   ├── impact_mapper.py # Stocks, correlations, PM markets
│       │   │   └── mispricing.py   # AI vs Market detection
│       │   │
│       │   ├── sentiment/      # FLOW 2: Sentiment
│       │   │   ├── __init__.py
│       │   │   ├── ticker.py       # Ticker extraction + blacklist
│       │   │   ├── aggregator.py   # Cross-platform sentiment
│       │   │   ├── divergence.py   # Divergence detection
│       │   │   ├── trending.py     # Velocity tracker + spike detection
│       │   │   └── watchlist.py    # Dynamic watchlist (auto-fed from Flow 1)
│       │   │
│       │   └── polymarket/     # FLOW 3: Polymarket Intel
│       │       ├── __init__.py
│       │       ├── onchain.py      # On-chain data fetcher
│       │       ├── wallet.py       # Wallet tracking
│       │       ├── insider.py      # Insider detection + scoring
│       │       └── cluster.py      # Cluster analysis (Louvain)
│       │
│       ├── decisions/          # Decision Layer
│       │   ├── __init__.py
│       │   ├── models.py           # TradeSignal, Alert, CorrelationSignal
│       │   ├── synthesizer.py      # Combines signals from flows
│       │   ├── confidence.py       # Confidence calculation
│       │   └── engine.py           # Main decision engine
│       │
│       ├── markets/            # Polymarket integration
│       │   ├── __init__.py
│       │   ├── gamma.py        # Gamma API client
│       │   ├── clob.py         # CLOB client wrapper
│       │   ├── websocket.py    # Real-time price feed
│       │   ├── matcher.py      # News → Market matching
│       │   └── router.py       # Market API routes
│       │
│       ├── trading/            # Strategy & execution
│       │   ├── __init__.py
│       │   ├── strategies/
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # Strategy interface
│       │   │   ├── news_driven.py  # LLM sentiment trading
│       │   │   ├── arbitrage.py    # Sum-to-one arb
│       │   │   ├── dutching.py     # Multi-outcome
│       │   │   ├── insider_copy.py # Copy high-score insiders
│       │   │   └── market_making.py # Spread capture
│       │   ├── executor.py     # Order execution
│       │   ├── risk.py         # Position limits, circuit breakers
│       │   └── router.py       # Trading API routes
│       │
│       ├── feedback/           # Learning from outcomes
│       │   ├── __init__.py
│       │   ├── tracker.py      # Outcome tracking
│       │   └── analytics.py    # Win rate & performance analytics
│       │
│       ├── storage/            # Database layer
│       │   ├── __init__.py
│       │   ├── database.py     # PostgreSQL + asyncpg
│       │   ├── redis.py        # Redis client
│       │   └── models.py       # Pydantic models
│       │
│       └── api/                # HTTP API
│           ├── __init__.py
│           ├── router.py       # Main router
│           ├── health.py       # Health checks
│           └── schemas.py      # API schemas
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── scripts/
│   ├── seed_accounts.py
│   ├── backfill_markets.py
│   └── run_backtest.py
│
└── docs/
    └── PRD2.md                 # This document
```

---

## 9. API Design

### 9.1 Core Endpoints

```
# Health & Status
GET  /health                    # Liveness check
GET  /ready                     # Readiness check
GET  /metrics                   # Prometheus metrics

# Ingestion
POST /api/v1/ingest/webhook     # Manual webhook trigger
GET  /api/v1/ingest/status      # Stream health

# Markets
GET  /api/v1/markets/search     # Search Polymarket
GET  /api/v1/markets/{id}       # Get market details
GET  /api/v1/markets/arbitrage  # Scan for arb opportunities

# Trading
GET  /api/v1/trading/signals    # Recent signals
GET  /api/v1/trading/positions  # Open positions
POST /api/v1/trading/execute    # Manual trade execution

# Analysis
GET  /api/v1/analysis/performance   # Strategy performance
GET  /api/v1/analysis/outcomes      # Signal outcomes
```

### 9.2 WebSocket Endpoints

```
WS /ws/signals      # Live trading signals
WS /ws/prices       # Market price updates
WS /ws/events       # Processed news events
```

---

## 10. Configuration

### 10.1 Environment Variables

```bash
# Core
SYNESIS_ENV=development|staging|production
SYNESIS_DEBUG=true|false
SYNESIS_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Telegram
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION_NAME=synesis
TELEGRAM_CHANNELS=marketfeed,disclosetv

# Twitter/X (via TwitterAPI.io - $0.15/1k tweets)
TWITTER_API_KEY=your_twitterapi_io_key
TWITTER_API_BASE_URL=https://api.twitterapi.io
TWITTER_ACCOUNTS=DeItaone,Newsquawk,FirstSquawk,zerohedge

# Reddit (via rss.app - $16.64/mo Developer plan)
RSS_APP_API_KEY=your_rss_app_api_key
REDDIT_RSS_FEEDS=https://rss.app/feeds/abc.xml,https://rss.app/feeds/def.xml
REDDIT_POLL_INTERVAL_SECONDS=300

# Polymarket
POLYMARKET_API_KEY=your_key
POLYMARKET_API_SECRET=your_secret
POLYMARKET_PRIVATE_KEY=your_wallet_pk
POLYMARKET_CHAIN_ID=137

# LLM (PydanticAI - model-agnostic, structured outputs)
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
LLM_MODEL=anthropic:claude-3-5-haiku-20241022

# Storage
DATABASE_URL=postgresql+asyncpg://synesis:synesis@localhost:5432/synesis
REDIS_URL=redis://localhost:6379

# Trading
TRADING_ENABLED=false
MAX_POSITION_SIZE=100
MIN_EDGE_THRESHOLD=0.05
CONFIDENCE_THRESHOLD=0.7
```

---

## 11. Risk Management

### 11.1 Circuit Breakers

| Trigger              | Action                   |
| -------------------- | ------------------------ |
| Daily loss > $X      | Halt all trading         |
| 5 consecutive losses | Reduce position size 50% |
| API errors > 10/min  | Pause ingestion          |
| Latency > 30s        | Alert + fallback         |

### 11.2 Position Limits

```python
class RiskManager:
    max_position_per_market: float = 100.0    # Max $100 per market
    max_total_exposure: float = 1000.0        # Max $1000 total
    max_daily_loss: float = 200.0             # Stop at $200 loss
    min_liquidity_ratio: float = 0.1          # Position < 10% of liquidity
```

---

## 12. Implementation Plan

### Phase 1: Flow 1 - Breaking News (Priority)

| Task | Description                | Files                                         |
| ---- | -------------------------- | --------------------------------------------- |
| 1.1  | Deduplication layer        | `processing/deduplication.py`                 |
| 1.2  | LLM Breaking News Analyzer | `intelligence/breaking_news/analyzer.py`      |
| 1.3  | Impact Mapper              | `intelligence/breaking_news/impact_mapper.py` |
| 1.4  | Mispricing Detector        | `intelligence/breaking_news/mispricing.py`    |
| 1.5  | Integration with ingestion | Update `ingestion/*.py`                       |

### Phase 2: Flow 2 - Sentiment Intelligence

| Task | Description                        | Files                                  |
| ---- | ---------------------------------- | -------------------------------------- |
| 2.1  | Reddit ingestion (PRAW)            | `ingestion/reddit.py`                  |
| 2.2  | Ticker extractor with blacklist    | `intelligence/sentiment/ticker.py`     |
| 2.3  | Sentiment aggregator               | `intelligence/sentiment/aggregator.py` |
| 2.4  | Cross-platform divergence detector | `intelligence/sentiment/divergence.py` |

### Phase 3: Flow 3 - Polymarket Intelligence

| Task | Description                     | Files                                |
| ---- | ------------------------------- | ------------------------------------ |
| 3.1  | On-chain data fetcher (Polygon) | `intelligence/polymarket/onchain.py` |
| 3.2  | Wallet tracker                  | `intelligence/polymarket/wallet.py`  |
| 3.3  | Insider detector with scoring   | `intelligence/polymarket/insider.py` |
| 3.4  | Cluster analyzer (Louvain)      | `intelligence/polymarket/cluster.py` |

### Phase 4: Decision Layer

| Task | Description                  | Files                      |
| ---- | ---------------------------- | -------------------------- |
| 4.1  | Decision models              | `decisions/models.py`      |
| 4.2  | Signal synthesizer           | `decisions/synthesizer.py` |
| 4.3  | Confidence calculator        | `decisions/confidence.py`  |
| 4.4  | Decision engine orchestrator | `decisions/engine.py`      |

### Verification Criteria

**Phase 1 Complete When:**

- [ ] Breaking news flows through analyzer → impact mapper
- [ ] Stocks and correlations correctly identified
- [ ] Polymarket markets matched and mispricing detected
- [ ] End-to-end test: Tweet → TradeSignal

**Phase 2 Complete When:**

- [ ] Reddit streaming operational
- [ ] Tickers correctly extracted (blacklist working)
- [ ] Cross-platform sentiment aggregated
- [ ] Divergence detection working

**Phase 3 Complete When:**

- [ ] On-chain trade data fetching works
- [ ] Insider scoring produces reasonable scores
- [ ] Cluster detection finds coordinated wallets

**Phase 4 Complete When:**

- [ ] Decision engine synthesizes all 3 flows
- [ ] Confidence bonuses for multi-flow agreement
- [ ] TradeSignals, Alerts, CorrelationSignals generated
- [ ] Integration with existing execution layer

---

## Appendix A: Polymarket Intelligence

### A.1 Power Users to Follow

#### Tier 1: Verified Alpha

| Handle           | Focus            | Why Follow                        |
| ---------------- | ---------------- | --------------------------------- |
| @thejayden       | Tools/Analysis   | Whale Watcher, latency strategies |
| @CarOnPolymarket | Trading/Building | Top 0.0001% trader                |
| @nicoco89poly    | Trading          | $410 → top 0.1%, $80K+ PnL        |
| @mombil          | Trading          | +$350K PnL from $10K deposit      |
| @cashyPoly       | Geopolitics      | Top 0.01%, Middle East specialist |
| @MomentumKevin   | Market Making    | PM MM, Top 500 Hyperliquid        |

#### Tier 2: Tool Builders

| Handle          | Tool             | Notes             |
| --------------- | ---------------- | ----------------- |
| @0xdotdot       | @nexustoolsfun   | Pro terminal      |
| @NevuaMarkets   | Alerts           | TG/Discord alerts |
| @zharkov_crypto | @PolyHuntApp     | Web3 dev          |
| @whalewatchpoly | mobyscreener.com | Whale tracking    |

### A.2 Notable Traders

| Profile             | Stats                       | Strategy                 |
| ------------------- | --------------------------- | ------------------------ |
| @kch123             | $6M all-time, $3.1M/month   | High volume              |
| "French Whale" Theo | $85M Trump election         | Political specialization |
| beachboy4           | +$5.7M (4 trades, Jan 2026) | Sports specialist        |

### A.3 Insider Detection Patterns

| Pattern                | Detection Signal            |
| ---------------------- | --------------------------- |
| Fresh Wallet + Big Bet | Account <24h + bet >$10k    |
| Single Market Focus    | Market diversity = 1        |
| Perfect Win Rate       | 100% in specific category   |
| Hidden Funding         | No on-chain deposit history |
| Wash Trading Cover     | Buy/sell same market        |

### A.4 Tools

**Analysis Platforms:**

- polymarketanalytics.com - Trader leaderboards
- polytrackhq.app - Follow top traders
- polywhaler.com - Whale tracking

**Telegram Bots:**

- t.me/PredictionIns - PolyInsiderBot
- @PolyxBot - Advanced TG trading
- @VeloriAIPredict - AI predictions

---

## Appendix B: Reddit Integration (via RSS)

### B.1 Why RSS Instead of Reddit API

| Approach               | Pros                          | Cons                                           |
| ---------------------- | ----------------------------- | ---------------------------------------------- |
| **Reddit API (PRAW)**  | Real-time, rich metadata      | Requires OAuth, 60 req/min limit, complex auth |
| **Native RSS (.rss)**  | No auth needed                | 429 errors, rate limited by Reddit             |
| **rss.app**            | Reliable, handles rate limits | $16.64/mo, 15-min refresh max                  |
| **Self-hosted Redlib** | Free, fast                    | Infrastructure overhead, maintenance           |

**Decision:** Use rss.app for simplicity and reliability. Reddit is for sentiment (not speed), so 15-min latency is acceptable.

### B.2 rss.app Setup

**Step 1: Create Feeds**

1. Go to [rss.app](https://rss.app)
2. Paste subreddit URL: `https://reddit.com/r/wallstreetbets/new`
3. Click Generate (takes ~20 seconds)
4. Save the feed URL: `https://rss.app/feeds/xxxxx.xml`

**Step 2: Recommended Feeds**

| Subreddit        | Sort | Feed Purpose                     |
| ---------------- | ---- | -------------------------------- |
| r/wallstreetbets | /new | Early sentiment, ticker mentions |
| r/wallstreetbets | /hot | Viral posts, high engagement     |
| r/stocks         | /new | General market sentiment         |
| r/options        | /new | Options flow discussion          |
| r/CryptoCurrency | /new | Crypto sentiment                 |
| r/polymarket     | /new | PM-specific discussion           |

**Step 3: Configure Alerts (Optional)**
rss.app supports sending alerts to Discord/Telegram/Slack when new posts arrive. This can supplement polling.

### B.3 RSS Parsing Implementation

**Library Choice:** `fastfeedparser` (10-50x faster than `feedparser`)

```python
import asyncio
import httpx
import fastfeedparser
from datetime import datetime, timezone
from dataclasses import dataclass

@dataclass
class RedditPost:
    id: str
    title: str
    body: str
    author: str
    subreddit: str
    url: str
    published: datetime
    score: int | None = None  # Not always in RSS

class RedditRSSPoller:
    """Poll rss.app feeds for Reddit content."""

    def __init__(self, feed_urls: list[str], poll_interval: int = 300):
        self.feed_urls = feed_urls
        self.poll_interval = poll_interval
        self.seen_ids: set[str] = set()
        self.client = httpx.AsyncClient(timeout=30.0)

    async def poll_feed(self, feed_url: str) -> list[RedditPost]:
        """Fetch and parse a single RSS feed."""
        response = await self.client.get(feed_url)
        response.raise_for_status()

        feed = fastfeedparser.parse(response.text)
        posts = []

        for entry in feed.entries:
            post_id = self._extract_id(entry)

            if post_id in self.seen_ids:
                continue

            self.seen_ids.add(post_id)

            posts.append(RedditPost(
                id=post_id,
                title=entry.get("title", ""),
                body=self._extract_body(entry),
                author=self._extract_author(entry),
                subreddit=self._extract_subreddit(entry),
                url=entry.get("link", ""),
                published=self._parse_date(entry.get("published")),
            ))

        return posts

    async def poll_all(self) -> list[RedditPost]:
        """Poll all feeds concurrently."""
        tasks = [self.poll_feed(url) for url in self.feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_posts = []
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
            # Log exceptions but don't fail

        return all_posts

    def _extract_id(self, entry: dict) -> str:
        """Extract unique ID from entry."""
        # rss.app typically uses guid or link
        return entry.get("id") or entry.get("link", "")

    def _extract_body(self, entry: dict) -> str:
        """Extract post body, handling HTML content."""
        content = entry.get("content", [{}])
        if isinstance(content, list) and content:
            return content[0].get("value", "")
        return entry.get("summary", "")

    def _extract_author(self, entry: dict) -> str:
        """Extract author from entry."""
        author = entry.get("author", "")
        if isinstance(author, dict):
            return author.get("name", "unknown")
        return author or "unknown"

    def _extract_subreddit(self, entry: dict) -> str:
        """Extract subreddit from link URL."""
        link = entry.get("link", "")
        # Parse /r/subreddit from URL
        if "/r/" in link:
            parts = link.split("/r/")[1].split("/")
            return parts[0] if parts else "unknown"
        return "unknown"

    def _parse_date(self, date_str: str | None) -> datetime:
        """Parse date with fallback."""
        if not date_str:
            return datetime.now(timezone.utc)
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except Exception:
            return datetime.now(timezone.utc)
```

### B.4 Handling Messy RSS Data

RSS feeds are notoriously inconsistent. Key defensive patterns:

```python
def safe_get(entry: dict, *keys, default=""):
    """Safely traverse nested dict with fallback."""
    value = entry
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        elif isinstance(value, list) and value:
            value = value[0] if key == 0 else default
        else:
            return default
    return value or default

# Example: entry.content[0].value with fallback
body = safe_get(entry, "content", 0, "value", default="")
```

**Common Issues:**

| Issue           | Example       | Solution                                 |
| --------------- | ------------- | ---------------------------------------- |
| Missing fields  | No `author`   | Default to "unknown"                     |
| HTML in content | `<p>text</p>` | BeautifulSoup `.get_text()`              |
| Malformed dates | "2 hours ago" | `dateutil.parser.parse()`                |
| Encoding errors | `\x00` bytes  | `entry.encode('utf-8', errors='ignore')` |
| Empty entries   | `{}`          | Skip with validation                     |

### B.5 Ticker Extraction (Unchanged)

```python
BLACKLIST = {
    # WSB slang
    "YOLO", "FOMO", "DD", "HODL", "BTFD", "FD", "ROPE", "MOON",
    "APE", "APES", "GAIN", "LOSS", "PUTS", "CALL", "CALLS",
    # Business terms
    "CEO", "CFO", "CTO", "COO", "IPO", "ATH", "ATL", "EPS",
    "GDP", "SEC", "FDA", "ETF", "NYSE", "NASDAQ", "OTC",
    # Common words
    "USA", "USD", "EUR", "GBP", "THE", "FOR", "AND", "ARE",
    "NOT", "YOU", "ALL", "CAN", "HAD", "HER", "WAS", "ONE",
    "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
    # Internet slang
    "LOL", "WTF", "OMG", "IMO", "TLDR", "LMAO", "IMHO",
    "TBH", "IDK", "SMH", "FYI", "BTW", "RN", "TIL",
    # Crypto false positives
    "DEFI", "NFT", "DAO", "DEX", "CEX", "KYC", "AML",
}
```

### B.6 Quality Scoring

Since RSS doesn't include Reddit score/upvote_ratio, we adjust scoring:

```python
WEIGHTS = {
    "title_sentiment": 0.30,      # LLM sentiment of title
    "body_length": 0.15,          # Longer = more substance
    "ticker_count": 0.20,         # More tickers = more tradeable
    "subreddit_relevance": 0.20,  # WSB > stocks > crypto
    "recency": 0.15,              # Newer = fresher signal
}
```

### B.7 Latency Budget

| Stage                  | Time                 |
| ---------------------- | -------------------- |
| Reddit post created    | t=0                  |
| rss.app scrapes Reddit | t+0 to t+15min       |
| rss.app feed updated   | t+15min (worst case) |
| Synesis polls rss.app  | t+15min to t+20min   |
| Processing + LLM       | t+20min to t+21min   |
| **Total**              | **15-21 minutes**    |

This is acceptable for **sentiment confirmation**, not breaking news. Twitter handles speed.

---

## Appendix C: Twitter/X Integration

### C.1 Primary Sources

```python
PRIMARY_SOURCES = {
    "journalists": ["DeItaone", "Newsquawk", "FirstSquawk", "zabormeister"],
    "crypto": ["WatcherGuru", "whale_alert", "lookonchain"],
    "prediction_markets": ["PolyAlertHub", "DonOfDAOs", "unusual_whales"],
    "official": ["WhiteHouse", "federalreserve", "SECGov"],
}
```

### C.2 Breaking News Detection

```python
BREAKING_INDICATORS = [
    "BREAKING", "JUST IN", "ALERT", "URGENT",
    "NOW:", "DEVELOPING", "CONFIRMED"
]
```

### C.3 Cashtag Monitoring

Monitor cashtag volume for early signals:

- Twitter processes 4.7 million cashtag searches daily
- 3x baseline velocity = spike detection
- Coordinated spike across multiple tickers = potential catalyst

---

## Future: PRD3 (Feedback & Learning)

_Not in scope for PRD2, but architecture supports:_

- Outcome tracking per signal
- LLM daily/weekly reviews
- Auto-adjustment of source weights
- Pattern learning

---

## References

### Documentation

- [Polymarket Developer Docs](https://docs.polymarket.com/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [Telethon Documentation](https://docs.telethon.dev/)
- [uv Project Guide](https://docs.astral.sh/uv/guides/projects/)
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [TwitterAPI.io Documentation](https://twitterapi.io/)
- [rss.app Help Center](https://help.rss.app/)
- [rss.app Reddit Feeds Guide](https://help.rss.app/en/articles/11164111-how-to-create-rss-feeds-from-reddit-subreddits)
- [FastFeedParser (PyPI)](https://pypi.org/project/fastfeedparser/)

### Research Sources

- [News-Driven Polymarket Bots - QuantVPS](https://www.quantvps.com/blog/news-driven-polymarket-bots)
- [Twitter API Alternatives Comparison](https://twitterapi.io/blog/twitter-api-alternatives-comprehensive-guide-2025)
- [Context Analytics: Multi-Source Sentiment 2025](https://www.contextanalytics-ai.com/sentiment-strategies/)
- [Stanford: Paper Trading from Sentiment Analysis](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/)
- [Alpaca: Reddit Sentiment Trading Strategy](https://alpaca.markets/learn/reddit-sentiment-analysis-trading-strategy)
- [Volume Spike Detection Research - SliceMatrix](https://slicematrix.github.io/stock_market_anomalies.html)
- [TwitterAPI.io Advanced Search Docs](https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search)
- [Fintwit.ai - Real-time FinTwit Analysis](https://fintwit.ai/)
- [Cashtag Collision Research - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417419301812)

---

_Last updated: 2026-01-26_
