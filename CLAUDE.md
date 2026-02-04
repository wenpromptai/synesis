# Synesis

## Purpose

Real-time financial news analysis and prediction market trading system. Transforms social signals (X/Twitter, Telegram) into actionable Polymarket trading decisions using LLM-powered analysis, market matching, and automated execution.

## Tech Stack

- Language: Python 3.12+
- Framework: FastAPI 0.115+
- Package manager: uv
- Database: PostgreSQL 16 + TimescaleDB + pgvector
- Cache/Queue: Redis
- LLM: Claude/OpenAI for analysis
- Trading: Polymarket CLOB API

## Commands

```bash
# Lint & Format
uv run ruff check --fix . && uv run ruff format .

# Type Check
uv run mypy src/

# Test
uv run pytest

# Run (production — server + agent in one process)
uv run synesis

# Run (development — auto-reload on file save)
uv run synesis --reload

# Docker services (PostgreSQL, Redis)
docker compose up -d
```

## Project Structure

```
src/synesis/
├── core/              # Logging, constants, dependencies
├── ingestion/         # Twitter, Telegram, Reddit listeners
├── processing/        # All analysis pipelines
│   ├── news/          # Flow 1: LLM news analysis (Stage 1 + Stage 2)
│   ├── sentiment/     # Flow 2: Reddit sentiment (Gate 1 lexicon + Gate 2 LLM)
│   └── common/        # Shared utilities (watchlist, LLM, web search)
├── providers/         # External data (Finnhub, FactSet, Crawl4AI)
├── markets/           # Polymarket integration
├── notifications/     # Telegram notifications
├── storage/           # PostgreSQL + Redis clients
├── agent/             # Agent runner, lifespan, PydanticAI
└── api/               # HTTP/WebSocket endpoints

tests/                 # Test files
docs/                  # Documentation (Obsidian vault)
scripts/               # Utility scripts
```

## Boundaries

**Never:**

- Commit secrets or .env files
- Execute trades without TRADING_ENABLED=true
- Delete files without asking
- Force push to main

**Ask first:**

- Adding new trading strategies
- Changing risk management parameters
- Refactors touching >5 files
- Adding new dependencies
- Changing CI/CD configs

## Context

- `docs/PRD.md` - Product requirements document
- `docs/Architecture/` - System architecture documentation
- `docs/Sources/` - Data source documentation (X accounts, Telegram)
- `docs/Implementation/` - Technical implementation details
- `.claude/skills/fastapi-developing/` - FastAPI patterns

## Key APIs

- **Polymarket Gamma API**: `https://gamma-api.polymarket.com` (market discovery)
- **Polymarket CLOB API**: `https://clob.polymarket.com` (trading)
- **Polymarket WebSocket**: `wss://ws-subscriptions-clob.polymarket.com/ws/market`

## Trading Strategies

1. **News-Driven Sentiment**: LLM analyzes news → finds related markets → trades
2. **Sum-to-One Arbitrage**: Buy YES + NO when total < $1.00
3. **Multi-Outcome Dutching**: Buy all outcomes when sum < $1.00
4. **Market Making**: Capture bid-ask spreads (optional)

See `docs/PRD.md` for full strategy documentation.
