# Synesis

Real-time financial news analysis and prediction market trading system. Transforms social signals (Telegram, Twitter) into actionable Polymarket trading decisions using LLM-powered analysis.

## Tech Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI 0.115+
- **Package Manager:** uv
- **Database:** PostgreSQL 16 + TimescaleDB
- **Cache/Queue:** Redis
- **LLM:** PydanticAI (Claude / OpenAI)
- **Data Providers:** SEC EDGAR, NASDAQ, Finnhub
- **Trading:** Polymarket (Gamma API + CLOB API)

## Quickstart

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
uv sync

# Run (production - server + agent in one process)
uv run synesis

# Run (development - auto-reload on file save)
uv run synesis --reload
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
├── notifications/     # Telegram notifications
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
