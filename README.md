# Synesis

Financial intelligence and prediction market research system. Runs LangGraph multi-agent pipelines to analyze equities, form bull/bear theses, and produce structured trade ideas. Monitors market events, Twitter signals, and market movers. Sends briefs to Discord.

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

```bash
# 1. Install dependencies
uv sync

# 2. Start infrastructure (DB, Redis, SearXNG, Crawl4AI)
docker compose up -d timescaledb redis searxng crawl4ai

# 3. Run app locally to verify startup
uv run synesis

# 4. Start the full stack
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
├── ingestion/         # Twitter client, price data
├── processing/        # All analysis pipelines
│   ├── intelligence/  # LangGraph multi-agent pipeline (on-demand via /analyze)
│   ├── twitter/       # Twitter agent: daily digest (LLM analysis + watchlist)
│   ├── market/        # Market movers: daily snapshot + top movers
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
├── notifications/     # Discord notifications
├── storage/           # PostgreSQL + Redis clients
├── agent/             # Scheduler, lifespan
└── api/               # HTTP/WebSocket endpoints
```

## Intelligence Pipeline

On-demand LangGraph state machine (`POST /api/v1/intelligence/analyze`):

```
ticker research + company/price analysis (parallel, per ticker)
→ bull/bear debate → Trader (equity R/R + conviction tiers)
→ Discord brief + KG markdown
```

Scheduled jobs:
- **10:00 AM ET** — Twitter agent digest → Discord
- **10:30 AM ET** — Market movers snapshot → Discord
- **6:00 PM ET** — Event Radar fetch
- **7:00 PM ET** — Event Radar digest → Discord

Each pipeline run auto-saves a markdown brief to `docs/kg/raw/synesis_briefs/` for knowledge graph compilation. See `docs/ARCHITECTURE.md` for full details.

## Knowledge Graph (`docs/kg/`)

LLM-compiled investment knowledge base (Karpathy-style). Raw sources are compiled into interlinked Obsidian markdown nodes. View in Obsidian.

```
docs/kg/
├── _index.md              # Master index (LLM reads this first)
├── _compile_log.md        # Audit trail
├── raw/                   # Source documents (PDFs, articles, pipeline briefs)
│   └── synesis_briefs/    # Auto-saved pipeline briefs
├── tickers/               # Per-ticker research dossiers (living files)
├── themes/                # Cross-cutting risk/opportunity themes
├── sources/               # Extracted summaries from raw documents
│   └── connections/       # Non-obvious relationships between nodes
├── maps/                  # Topic indexes (MOCs)
├── concepts/              # Atomic concept nodes
└── strategies/            # Strategy playbooks
```

**Claude Code commands:**
- `/daily-brief` — Claude Code-powered intelligence brief (complements the pipeline)
- `/kg-compile` — Compile unprocessed raw files into KG nodes (run after new sources accumulate)
- `/kg-lint` — Health checks + intelligence checks (run periodically)

## Development

```bash
# Lint & format
uv run ruff check --fix . && uv run ruff format .

# Type check (strict)
uv run mypy src/

# Run tests
uv run pytest
```
