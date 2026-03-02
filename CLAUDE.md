# Synesis

## Purpose

Real-time financial news analysis and prediction market trading system. Transforms social signals (Telegram, Twitter) into actionable Polymarket trading decisions using LLM-powered analysis, market matching, and automated execution.

## Tech Stack

- Language: Python 3.12+
- Framework: FastAPI 0.115+
- Package manager: uv
- Database: PostgreSQL 16 + TimescaleDB
- Cache/Queue: Redis
- LLM: PydanticAI (Claude / OpenAI)
- Trading: Polymarket (Gamma API for discovery, CLOB API for execution)

## Commands

```bash
# Lint & Format
uv run ruff check --fix . && uv run ruff format .

# Type Check
uv run mypy src/

# Test
uv run pytest

# Run locally (development — auto-reload on file save)
uv run synesis --reload
```

### Docker Setup (first time)

Telegram requires an interactive login on first run, so the app cannot start in Docker until a session file exists.

```bash
# 1. Start infrastructure only (DB, Redis, SearXNG, Crawl4AI)
docker compose up -d timescaledb redis searxng crawl4ai

# 2. Run app locally to generate Telegram session (shared/sessions/synesis.session)
uv run synesis
# Wait for "Signed in successfully", then Ctrl+C

# 3. Now start the full stack including the app
docker compose up -d
```

### Docker (already set up)

```bash
# Start everything
docker compose up -d

# Restart app only (e.g. after code changes)
docker compose build synesis && docker compose up -d synesis

# View app logs
docker compose logs -f synesis
```

## Project Structure

```
src/synesis/
├── core/              # Logging, constants, dependencies
├── ingestion/         # Telegram, Twitter listeners
├── processing/        # All analysis pipelines
│   ├── news/          # Flow 1: LLM news analysis (Stage 1 + Stage 2)
│   └── common/        # Shared utilities (watchlist, LLM, web search)
├── providers/         # External data providers
│   ├── finnhub/       # Real-time prices, fundamentals
│   ├── nasdaq/        # NASDAQ earnings calendar
│   ├── sec_edgar/     # SEC EDGAR filings, insider transactions, earnings
│   └── crawler/       # Crawl4AI HTML-to-markdown (Docker service)
├── markets/           # Polymarket integration
├── notifications/     # Telegram & Discord notifications
├── storage/           # PostgreSQL + Redis clients
├── agent/             # Agent runner, scheduler, lifespan, PydanticAI
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

- `.claude/skills/fastapi-developing/` - FastAPI patterns

## Key APIs

- **Polymarket Gamma API**: `https://gamma-api.polymarket.com` (market discovery)
- **SEC EDGAR API**: `https://data.sec.gov` (filings, Form 4, XBRL — free, no key)
- **NASDAQ Earnings**: `https://api.nasdaq.com` (earnings calendar — free, no key)

## API Routes

All routes are mounted under `/api/v1/`:

- `/fh_prices/*` — Finnhub real-time prices (WebSocket management)
- `/sec_edgar/*` — SEC filings, insider transactions, insider sentiment, earnings content
- `/earnings/*` — NASDAQ earnings calendar
- `/watchlist/*` — Watchlist management
- `/system/*` — System status

## Trading Strategy

1. **News-Driven Sentiment**: LLM analyzes news → finds related Polymarket markets → evaluates mispricing
