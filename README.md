# Synesis

Real-time financial news analysis and prediction market trading system. Transforms social signals (Telegram, Google News RSS) into actionable Polymarket trading decisions using LLM-powered analysis.

## Tech Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI 0.115+
- **Package Manager:** uv
- **Database:** PostgreSQL 16 + TimescaleDB
- **Cache/Queue:** Redis
- **LLM:** PydanticAI (Claude / OpenAI)
- **Data Providers:** SEC EDGAR, NASDAQ, Finnhub, yfinance, FRED, Massive.com
- **Intelligence:** LangGraph multi-agent pipeline (PydanticAI)
- **Knowledge:** LLM-compiled knowledge graph (Obsidian markdown)
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
├── ingestion/         # Telegram listener, Google News RSS poller
├── processing/        # All analysis pipelines
│   ├── intelligence/  # LangGraph multi-agent pipeline (daily briefs)
│   ├── news/          # Flow 1: impact scoring + ticker matching → LLM analysis
│   ├── twitter/       # Twitter agent: daily digest (LLM analysis + watchlist)
│   ├── market/        # Market movers: daily snapshot + top movers + diary
│   ├── events/        # Event radar: forward-looking calendar digest
│   └── common/        # Shared utilities (watchlist, LLM, web search)
├── providers/         # External data providers
│   ├── finnhub/       # Real-time prices, fundamentals
│   ├── nasdaq/        # NASDAQ earnings calendar
│   ├── sec_edgar/     # SEC EDGAR filings, insiders, 13F holdings, ownership, XBRL
│   ├── yfinance/      # Equity/ETF/FX quotes, OHLCV history, options chains, analyst ratings
│   ├── fred/          # FRED economic data (series, releases, observations)
│   ├── massive/       # Massive.com stocks + options (free tier, Polygon-compatible)
│   └── crawler/       # Crawl4AI HTML-to-markdown (Docker service)
├── markets/           # Polymarket integration
├── notifications/     # Telegram & Discord notifications
├── storage/           # PostgreSQL + Redis clients
├── agent/             # Agent runner, scheduler, lifespan, PydanticAI
└── api/               # HTTP/WebSocket endpoints
```

## Intelligence Pipeline

Daily LangGraph state machine (9:00 AM SGT / 1:00 AM UTC) that transforms social/news signals into structured trade ideas:

```
social_sentiment + news_analyst → extract_tickers → company + price + macro (parallel)
→ bull/bear debate (configurable rounds) → Trader → compiler → Discord + KG brief
```

Each pipeline run auto-saves a markdown brief to `docs/kg/raw/synesis_briefs/` for knowledge graph compilation. See `docs/ARCHITECTURE.md` for full details.

## Knowledge Graph (`docs/kg/`)

LLM-compiled investment knowledge base (Karpathy-style). Raw sources are compiled into interlinked Obsidian markdown nodes. View in Obsidian.

```
docs/kg/
├── _index.md              # Master index (LLM reads this first)
├── _compile_log.md        # Audit trail
├── raw/                   # Source documents (PDFs, articles, pipeline briefs)
│   └── synesis_briefs/    # Auto-saved daily pipeline briefs
├── tickers/               # Per-ticker research dossiers (living files)
├── themes/                # Cross-cutting risk/opportunity themes
├── sources/               # Extracted summaries from raw documents
│   └── connections/       # Non-obvious relationships between nodes
├── maps/                  # Topic indexes (MOCs)
├── concepts/              # Atomic concept nodes
└── strategies/            # Strategy playbooks
```

**Claude Code commands:**
- `/daily-brief` — Claude Code-powered intelligence brief (replaces/complements the pipeline)
- `/kg-compile` — Compile unprocessed raw files into KG nodes (run after new sources accumulate)
- `/kg-lint` — Health checks + intelligence checks (run periodically)

## Development

```bash
# Lint & format
uv run ruff check --fix . && uv run ruff format .

# Type check
uv run mypy src/

# Run tests
uv run pytest
```
