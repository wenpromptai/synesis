# Product Requirements Document: Synesis Trading Backend

**Status:** Draft
**Created:** 2026-01-17
**Last Updated:** 2026-01-17

---

## Executive Summary

Synesis is a real-time financial news analysis and prediction market trading system. This PRD defines the Python backend that transforms social signals (X/Twitter, Telegram) into actionable Polymarket trading decisions using LLM-powered analysis, market matching, and automated execution.

---

## 1. Vision & Goals

### 1.1 Product Vision

Build an autonomous trading system that:
1. **Ingests** real-time financial news from X/Twitter and Telegram
2. **Analyzes** content using LLMs for sentiment, entities, and market impact
3. **Matches** news to relevant Polymarket prediction markets
4. **Executes** reasoned bets based on configurable strategies
5. **Tracks** outcomes and calculates win rates by source/event type

### 1.2 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| News-to-signal latency | < 5 seconds | Time from ingestion to signal generation |
| Market match accuracy | > 80% | Relevant market found for news events |
| Signal profitability | > 55% win rate | Directional accuracy on tracked signals |
| System uptime | > 99.5% | Backend availability |

### 1.3 Non-Goals (Phase 1)

- Frontend/dashboard (future phase)
- Multi-exchange arbitrage (Kalshi integration deferred)
- High-frequency trading (< 100ms latency)
- Automated position sizing/Kelly criterion

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SYNESIS BACKEND                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         INGESTION LAYER                               │   │
│  │                                                                       │   │
│  │   ┌─────────────┐          ┌─────────────┐                           │   │
│  │   │  Telegram   │          │  X/Twitter  │                           │   │
│  │   │  Listener   │          │   Stream    │                           │   │
│  │   │ (Telethon)  │          │ (httpx/v2)  │                           │   │
│  │   └──────┬──────┘          └──────┬──────┘                           │   │
│  │          │                        │                                   │   │
│  │          └───────────┬────────────┘                                   │   │
│  │                      ▼                                                │   │
│  │          ┌───────────────────────┐                                   │   │
│  │          │   Unified Message     │                                   │   │
│  │          │       Queue           │                                   │   │
│  │          │   (Redis Streams)     │                                   │   │
│  │          └───────────┬───────────┘                                   │   │
│  └──────────────────────┼───────────────────────────────────────────────┘   │
│                         ▼                                                    │
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

### 2.2 Cross-Source Deduplication

The deduplicator is **unified across Telegram and Twitter**. When the same news event arrives from multiple sources:

1. **First source wins** - Gets full LLM analysis and signal generation
2. **Subsequent sources** - Logged as "coverage" for the same event
3. **Semantic matching** - Catches paraphrases, not just exact text matches

```
Example:
  12:00:01 - Telegram @marketfeed: "FED CUTS RATES BY 25BPS"
  12:00:03 - Twitter @DeItaone: "BREAKING: Federal Reserve cuts interest rates by 0.25%"

  Result:
  - Event ID: evt_abc123
  - Primary source: Telegram @marketfeed (processed)
  - Coverage: Twitter @DeItaone (logged, not re-processed)
```

### 2.3 Component Responsibilities

| Component | Responsibility | Key Dependencies |
|-----------|---------------|------------------|
| **Telegram Listener** | Real-time message streaming from channels | Telethon |
| **Twitter Stream** | Filtered tweet streaming from accounts | httpx (Filtered Stream v2) |
| **Unified Queue** | Merge messages from both sources | Redis Streams |
| **Cross-Source Deduplicator** | Semantic dedup across Telegram + Twitter | semhash, model2vec, pgvector |
| **LLM Analyzer** | Sentiment, classification, entity extraction | anthropic, openai |
| **Market Matcher** | Find relevant Polymarket markets | Gamma API |
| **Strategy Engine** | Apply trading strategies to signals | Custom logic |
| **Order Executor** | Place/cancel orders on CLOB | py-clob-client |
| **WebSocket Feed** | Real-time price/orderbook updates | websockets |
| **Outcome Tracker** | Track market resolutions, calculate win rates | Scheduler, PostgreSQL |

---

## 3. Tech Stack

### 3.1 Core Technologies

| Category | Technology | Version | Rationale |
|----------|------------|---------|-----------|
| **Language** | Python | 3.12+ | Async support, ecosystem |
| **Package Manager** | uv | latest | 10-100x faster than pip |
| **Web Framework** | FastAPI | 0.115+ | Async-first, type hints |
| **Task Queue** | Celery + Redis | 5.4+ | Distributed tasks |
| **Message Queue** | Redis Streams | 7+ | Lightweight pub/sub |

### 3.2 Data & Storage

| Category | Technology | Purpose |
|----------|------------|---------|
| **Database** | PostgreSQL 16 + TimescaleDB + pgvector | All data (relational, time-series, vectors) |
| **DB Driver** | asyncpg (raw SQL) | Direct queries for speed, no ORM overhead |
| **Cache/Queue** | Redis | Message queue, rate limits, cache |
| **Migrations** | SQL files in `database/` | Simple, version-controlled schema changes |

### 3.3 External APIs

| Service | SDK/Library | Purpose |
|---------|-------------|---------|
| **Polymarket** | py-clob-client | Trading, orderbook |
| **Polymarket** | Gamma API (REST) | Market discovery |
| **Polymarket** | WebSocket | Real-time prices |
| **Telegram** | Telethon | Channel streaming |
| **X/Twitter** | TwitterAPI.io (httpx) | Tweet streaming (3rd-party, $0.15/1k tweets) |
| **LLM** | PydanticAI | Structured analysis (model-agnostic: Claude, OpenAI, etc.) |
| **Embeddings** | PydanticAI / pgvector | Vector generation & similarity |

### 3.4 Infrastructure

| Category | Technology | Purpose |
|----------|------------|---------|
| **Containerization** | Docker | Development & deployment |
| **Orchestration** | Docker Compose | Local multi-service |
| **Monitoring** | Prometheus + Grafana | Metrics |
| **Logging** | structlog | Structured JSON logs |
| **Secrets** | python-dotenv | Environment variables |

---

## 4. Project Structure

```
synesis/
├── pyproject.toml              # Project config & dependencies
├── uv.lock                     # Locked dependencies
├── .env.example                # Environment template
├── .python-version             # Python version (3.12)
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
│       │   ├── twitter.py      # X/Twitter stream
│       │   ├── webhooks.py     # Webhook receivers
│       │   └── router.py       # Ingestion API routes
│       │
│       ├── processing/         # Analysis & enrichment
│       │   ├── __init__.py
│       │   ├── deduplication.py    # SemHash deduplicator
│       │   ├── classifier.py       # LLM classification
│       │   ├── entities.py         # Entity extraction
│       │   ├── embeddings.py       # Vector generation
│       │   └── pipeline.py         # Orchestrates processing
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
│       │   ├── database.py     # PostgreSQL + async SQLAlchemy
│       │   ├── redis.py        # Redis client
│       │   └── models.py       # SQLAlchemy + Pydantic models
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
│   │   ├── test_classifier.py
│   │   ├── test_matcher.py
│   │   └── test_strategies.py
│   ├── integration/
│   │   ├── test_telegram.py
│   │   ├── test_polymarket.py
│   │   └── test_pipeline.py
│   └── e2e/
│       └── test_full_flow.py
│
├── scripts/
│   ├── seed_accounts.py        # Seed X account configs
│   ├── backfill_markets.py     # Historical market data
│   └── run_backtest.py         # Strategy backtesting
│
├── migrations/                 # Alembic migrations
│   └── versions/
│
└── docs/                       # Documentation
    ├── PRD.md                  # This document
    └── ...
```

---

## 5. Trading Strategies

### 5.1 Strategy Overview

| Strategy | Type | Risk | Expected Edge |
|----------|------|------|---------------|
| **News-Driven Sentiment** | Directional | Medium | LLM analysis before market prices in |
| **Sum-to-One Arbitrage** | Risk-Free | None | YES + NO < $1.00 |
| **Multi-Outcome Dutching** | Risk-Free | None | All outcomes sum < $1.00 |
| **Conditional Correlation** | Low | Low | Related market mispricing |
| **Market Making** | Neutral | Medium | Bid-ask spread capture |

### 5.2 News-Driven Sentiment Strategy

```python
class NewsDrivenStrategy(BaseStrategy):
    """
    Trades Polymarket based on LLM analysis of breaking news.

    Flow:
    1. Receive analyzed news event with sentiment
    2. Search for related markets via Gamma API
    3. Calculate expected probability shift
    4. Execute if confidence > threshold and edge > 5%
    """

    async def evaluate(self, signal: Signal) -> Optional[Trade]:
        # Find related markets
        markets = await self.matcher.find_markets(
            keywords=signal.entities.keywords,
            categories=signal.classification.event_categories
        )

        if not markets:
            return None

        for market in markets:
            current_price = await self.clob.get_midpoint(market.yes_token_id)

            # Estimate probability shift from sentiment
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

### 5.3 Arbitrage Strategy

```python
class ArbitrageStrategy(BaseStrategy):
    """
    Detects and executes risk-free arbitrage when YES + NO < $1.00.

    Polymarket charges 2% fee on net winnings, so need > 2% margin.
    """

    async def scan_markets(self) -> List[ArbitrageOpportunity]:
        markets = await self.gamma.get_active_markets(
            liquidity_num_min=10000,  # Min $10k liquidity
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

---

## 6. Data Flow

### 6.1 News-to-Trade Pipeline

```
Telegram (@marketfeed)
        │
        ▼
   Ingestion Service
        │
        ▼
   Deduplicator (SemHash)
        │
        ├──[Duplicate]──▶ Log & Skip
        │
        ▼ [New Event]
   LLM Analyzer
   - Extract entities
   - Classify sentiment
   - Assess urgency
        │
        ▼
   Market Matcher
   - Gamma API search
   - Find related markets
        │
        ▼
   Strategy Engine
   - Evaluate strategies
   - Calculate edge
        │
        ├──[No Trade]──▶ Store signal only
        │
        ▼ [Actionable]
   Order Executor
   - Place CLOB order
   - Log execution
        │
        ▼
   Polymarket
```

### 6.2 Outcome Tracking Flow

```
Scheduler (hourly)
        │
        ▼
   Check pending signals
        │
        ▼
   Query Polymarket resolution
        │
        ├──[Not Resolved]──▶ Reschedule
        │
        ▼ [Resolved]
   Calculate P&L
        │
        ▼
   Store outcome in PostgreSQL
        │
        ▼
   Update win rate stats
   (by source, event_type, strategy)
```

---

## 7. API Design

### 7.1 Core Endpoints

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

### 7.2 WebSocket Endpoints

```
WS /ws/signals      # Live trading signals
WS /ws/prices       # Market price updates
WS /ws/events       # Processed news events
```

---

## 8. Configuration

### 8.1 Environment Variables

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
TWITTER_ACCOUNTS=DeItaone,KobeissiLetter,Fxhedgers,zerohedge

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

## 9. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Project scaffolding with uv + FastAPI
- [ ] Docker Compose for PostgreSQL (TimescaleDB + pgvector) and Redis
- [ ] Telegram listener with Telethon
- [ ] Basic LLM classification pipeline
- [ ] Polymarket Gamma API integration

### Phase 2: Trading Core (Weeks 3-4)
- [ ] CLOB client integration
- [ ] WebSocket price streaming
- [ ] Market matcher (news → markets)
- [ ] News-driven strategy implementation
- [ ] Order execution with risk limits

### Phase 3: Advanced Strategies (Weeks 5-6)
- [ ] Arbitrage scanner
- [ ] Multi-outcome dutching
- [ ] Market making (optional)
- [ ] Outcome tracking & win rate analytics

### Phase 4: Hardening (Weeks 7-8)
- [ ] Comprehensive test suite
- [ ] Monitoring & alerting
- [ ] Performance optimization
- [ ] Documentation

---

## 10. Risk Management

### 10.1 Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily loss > $X | Halt all trading |
| 5 consecutive losses | Reduce position size 50% |
| API errors > 10/min | Pause ingestion |
| Latency > 30s | Alert + fallback |

### 10.2 Position Limits

```python
class RiskManager:
    max_position_per_market: float = 100.0    # Max $100 per market
    max_total_exposure: float = 1000.0        # Max $1000 total
    max_daily_loss: float = 200.0             # Stop at $200 loss
    min_liquidity_ratio: float = 0.1          # Position < 10% of liquidity
```

---

## 11. Success Criteria

### Phase 1 Complete When
- [ ] Telegram messages flow through pipeline
- [ ] LLM correctly classifies sentiment
- [ ] Markets are matched to news events
- [ ] System runs stable for 24 hours

### MVP Complete When
- [ ] First automated trade executed
- [ ] Arbitrage opportunities detected
- [ ] Outcome tracking operational
- [ ] > 50% directional accuracy on tracked signals

---

## 12. Open Questions

- [x] ~~Twitter API tier (Basic $100/mo vs Pro $5000/mo)?~~ → Using TwitterAPI.io ($0.15/1k tweets)
- [x] ~~LLM framework?~~ → PydanticAI for structured outputs + model flexibility
- [x] ~~ORM vs raw SQL?~~ → Raw asyncpg for speed (no ORM overhead)
- [ ] Specific Telegram channels to prioritize?
- [ ] Specific X accounts to prioritize?
- [ ] Initial capital allocation per strategy?
- [ ] Deployment environment (local vs cloud)?

---

## 13. References

### Documentation
- [Polymarket Developer Docs](https://docs.polymarket.com/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [Telethon Documentation](https://docs.telethon.dev/)
- [uv Project Guide](https://docs.astral.sh/uv/guides/projects/)
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [TwitterAPI.io Documentation](https://twitterapi.io/)

### Research Sources
- [News-Driven Polymarket Bots - QuantVPS](https://www.quantvps.com/blog/news-driven-polymarket-bots)
- [Twitter API Alternatives Comparison](https://twitterapi.io/blog/twitter-api-alternatives-comprehensive-guide-2025)
- [@thejayden strategy thread](https://x.com/thejayden/status/2006276445409091692)
- [@dunik_7 GitHub repos thread](https://x.com/dunik_7/status/2004512366675829093)
