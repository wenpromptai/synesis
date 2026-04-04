# Synesis

Real-time financial news analysis and prediction market trading system. Transforms social signals (Telegram, Twitter) into actionable Polymarket trading decisions using LLM-powered analysis.

## Tech Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI 0.115+
- **Package Manager:** uv
- **Database:** PostgreSQL 16 + TimescaleDB
- **Cache/Queue:** Redis
- **LLM:** PydanticAI (Claude / OpenAI)
- **Data Providers:** SEC EDGAR, NASDAQ, Finnhub, yfinance, FRED, Massive.com
- **Trading:** Polymarket (Gamma API + CLOB API)

## Quickstart

### First-time setup

Telegram requires an interactive login on first run, so the app cannot start in Docker until a session file exists.

```bash
# 1. Install dependencies
uv sync

# 2. Start infrastructure only (DB, Redis, SearXNG, Crawl4AI)
docker compose up -d timescaledb redis searxng crawl4ai

# 3. Run app locally to generate Telegram session
uv run synesis
# Wait for "Signed in successfully", then Ctrl+C

# 4. Start the full stack including the app
docker compose up -d
```

### Already set up

```bash
# Start everything
docker compose up -d

# Restart app only (e.g. after code changes)
docker compose build synesis && docker compose up -d synesis

# View app logs
docker compose logs -f synesis

# Run locally with auto-reload (stop the Docker app first)
docker compose stop synesis && uv run synesis --reload
```

## Project Structure

```
src/synesis/
├── core/              # Logging, constants, dependencies
├── ingestion/         # Telegram, Twitter listeners
├── processing/        # All analysis pipelines
│   ├── news/          # Flow 1: impact scoring + ticker matching → LLM analysis
│   ├── twitter/       # Twitter agent: daily digest (LLM analysis + watchlist)
│   ├── market/        # Market brief: daily snapshot + LLM analysis + diary
│   ├── events/        # Event radar: daily digest
│   │   └── yesterday/ # Yesterday brief sub-analyzers (earnings, macro, surprises, filings, consolidator)
│   └── common/        # Shared utilities (watchlist, LLM, web search)
├── providers/         # External data providers
│   ├── finnhub/       # Real-time prices, fundamentals
│   ├── nasdaq/        # NASDAQ earnings calendar
│   ├── sec_edgar/     # SEC EDGAR filings, insiders, 13F holdings, ownership, XBRL
│   ├── yfinance/      # Equity/ETF/FX quotes, OHLCV history, options chains
│   ├── fred/          # FRED economic data (series, releases, observations)
│   ├── massive/       # Massive.com stocks + options (free tier, Polygon-compatible)
│   └── crawler/       # Crawl4AI HTML-to-markdown (Docker service)
├── markets/           # Polymarket integration
├── notifications/     # Telegram & Discord notifications
├── storage/           # PostgreSQL + Redis clients
├── agent/             # Agent runner, scheduler, lifespan, PydanticAI
└── api/               # HTTP/WebSocket endpoints
```

## Development

```bash
# Lint & format
uv run ruff check --fix . && uv run ruff format .

# Type check
uv run mypy src/

# Run tests
uv run pytest
```
