# Synesis

## Purpose

Financial intelligence and prediction market research system. Runs LangGraph multi-agent pipelines to analyze equities, form bull/bear theses, and produce structured trade ideas. Monitors market events, Twitter signals, and market movers. Sends briefs to Discord.

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

```bash
# 1. Start infrastructure only (DB, Redis, SearXNG, Crawl4AI)
docker compose up -d timescaledb redis searxng crawl4ai

# 2. Run app locally to verify startup
uv run synesis

# 3. Start the full stack including the app
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
├── ingestion/         # Twitter client, price data
├── processing/        # All analysis pipelines
│   ├── intelligence/  # LangGraph multi-agent pipeline (see docs/ARCHITECTURE.md)
│   │   ├── specialists/   # company, price, ticker_research analysts
│   │   ├── debate/        # Bull/bear debate subgraph (configurable rounds)
│   │   ├── trader/        # Trader (sole decision maker → equity TradeIdea with R/R)
│   │   ├── graph.py       # LangGraph state machine wiring
│   │   ├── compiler.py    # Brief assembly + markdown export for KG
│   │   └── job.py         # Pipeline runner → Discord + KG brief
│   ├── twitter/       # Twitter agent: daily digest (LLM analysis + watchlist)
│   ├── market/        # Market movers: daily snapshot + top movers
│   ├── events/        # Event radar: forward-looking calendar digest
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
├── notifications/     # Discord notifications
├── storage/           # PostgreSQL + Redis clients
├── agent/             # Scheduler, lifespan
└── api/               # HTTP/WebSocket endpoints

tests/                 # Test files
docs/                  # Documentation (Obsidian vault)
docs/kg/               # LLM-compiled knowledge graph (Karpathy-style)
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
- `.claude/skills/obsidian-kg/` - Knowledge graph building, compilation, and linting
- `docs/ARCHITECTURE.md` - Intelligence pipeline architecture (LangGraph topology, state, agents)

## Key APIs

- **Polymarket Gamma API**: `https://gamma-api.polymarket.com` (market discovery)
- **SEC EDGAR API**: `https://data.sec.gov` (filings, Form 4, XBRL — free, no key)
- **NASDAQ Earnings**: `https://api.nasdaq.com` (earnings calendar — free, no key)
- **FRED API**: `https://api.stlouisfed.org/fred` (economic data — free key required)
- **Massive.com API**: `https://api.massive.com` (stocks + options — free tier, 5 calls/min)

## API Routes

All routes are mounted under `/api/v1/`. Rate-limited via slowapi (per-IP).

- `/system/*` — System status (60/min)
- `/fh/*` — Finnhub: ticker verify/search (120/min), REST quotes (60/min), WS cache reads (120/min)
- `/yf/*` — yfinance: quotes, history, FX, options chains with Greeks, options snapshot, analyst ratings (30/min, chain/snapshot 10/min)
- `/watchlist/*` — Watchlist CRUD (reads 60/min, writes 10/min)
- `/earnings/*` — NASDAQ earnings calendar (30/min)
- `/sec_edgar/*` — SEC filings, insider transactions, sentiment, search (60/min, earnings content 10/min)
- `/fred/*` — FRED: series search, observations, releases (30/min, info 60/min)
- `/events/*` — Event radar: upcoming, calendar, discover, digest, CRUD (30/min, digest 5/min)
- `/twitter/*` — Twitter agent: trigger daily digest (5/min)
- `/market/*` — Market movers: trigger daily movers snapshot (5/min)
- `/intelligence/analyze` — On-demand deep ticker analysis (2/min)

See `src/synesis/api/routes/_routes_context.md` for full endpoint reference with examples.

## Intelligence Pipeline

On-demand via `/api/v1/intelligence/analyze`: ticker research + company/price analysis (parallel per ticker) → bull/bear debate → Trader (equity R/R + conviction tiers) → Discord brief + KG markdown.

Scheduled jobs (APScheduler):
- **10:00 AM ET** — Twitter agent digest
- **10:30 AM ET** — Market movers snapshot
- **6:00 PM ET** — Event Radar fetch
- **7:00 PM ET** — Event Radar digest → Discord
- **Monday 6am UTC** — Ticker list refresh

See `docs/ARCHITECTURE.md` for full graph topology, state schema, and agent inventory.

**Pipeline brief auto-save:** Each run saves a markdown brief to `docs/kg/raw/synesis_briefs/YYYY-MM-DD-tradeideas.md` for future KG compilation.

## Knowledge Graph (`docs/kg/`)

LLM-compiled investment knowledge base viewed in Obsidian. Raw sources (pipeline briefs, PDFs, articles) are compiled into interlinked ticker, theme, concept, strategy, source, and connection nodes.

**Slash commands:**
- `/daily-brief` — Claude Code-powered intelligence brief. Researches tickers via local API + web search, forms bull/bear views, produces trade ideas. Output saved to `docs/kg/raw/synesis_briefs/`.
- `/kg-compile` — Process uncompiled raw files in `docs/kg/raw/` into KG nodes. The LLM reads the schema + current KG state + raw source and decides what to extract/update/create. Run after new raw sources accumulate.
- `/kg-lint` — Health checks (broken links, orphans, sparse nodes, missing frontmatter, stale index) + intelligence checks (connection discovery, missing node candidates, content staleness, research suggestions). Run periodically to maintain KG quality and discover growth opportunities.

**Skill reference:** `.claude/skills/obsidian-kg/SKILL.md`

## Trading Strategy

1. **Intelligence Pipeline**: LangGraph agents analyze equities → bull/bear debate → Trader forms R/R trade ideas → Discord brief
