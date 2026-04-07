# Multi-Agent Intelligence Pipeline — PRD

## Vision

Upgrade Synesis from 4 independent pipelines into a LangGraph-orchestrated multi-agent system that produces a daily intelligence brief + actionable trade recommendations (stocks + options).

## Phases

### Phase 1: Pure Tweet Data Collection [DONE]
**Goal**: Twitter pipeline becomes pure data collection — fetch and store tweets, no LLM.

**Changes**:
- `raw_tweets` table with composite PK `(account_username, tweet_id)` — stores all fetched tweets, DB handles dedup
- Twitter job stripped to: fetch tweets from accounts → store to `raw_tweets` → done
- Deleted `analyzer.py` (LLM agent) and `models.py` (output types) — analysis moves to Phase 3 SocialSentimentAnalyst
- Removed Twitter Discord digest, watchlist auto-add, diary persistence
- `accounts.py` preserved — account profiles/biases reused in Phase 3

**Patterns preserved for Phase 3 SocialSentimentAnalyst**:
- Account bias/credibility weighting (accounts.py + get_profile())
- Web search + web_read tool pattern with budget caps
- verify_ticker tool
- Tweet formatting grouped by account with timestamps
- Cross-confirmation priority (themes from multiple accounts)
- From TradingAgents: mandatory structured table output, current date injection

**Files**: `database/init.sql`, `storage/database.py`, `twitter/job.py`, `agent/__main__.py`, `notifications/discord.py`, tests

---

### Phase 2: CompanyAnalyst + YFinance Extension [DONE]
**Goal**: Build a comprehensive US company analysis agent combining SEC EDGAR + yfinance data, using a deterministic-scoring-then-LLM-synthesis pattern proven by top implementations (virattt/ai-hedge-fund, MarketSenseAI 2.0).

**YFinance Client Extension**:
- `get_fundamentals(ticker)` → `CompanyFundamentals` model — pre-computed ratios from yfinance `.info`
- `get_quarterly_financials(ticker)` → `QuarterlyFinancials` model — 5 quarters of income, balance sheet, cash flow from yfinance (replaces XBRL for structured financial data — updates same-day vs XBRL which lags until 10-K/10-Q filing)
- Redis-cached with configurable TTL

**CompanyAnalyst** (OpenAI 5.2, 400k context):
- Three-phase pipeline per ticker:
  1. **Data Gathering** (no LLM): yfinance fundamentals + quarterly financials (replaces XBRL), SEC EDGAR insider transactions (Form 4, MSPR, Form 144), latest 10-K/10-Q filing prose via Crawl4AI
  2. **Deterministic Scoring** (no LLM): Piotroski F-Score (0-9) from yfinance quarterly data, insider cluster detection, red flag detection
  3. **LLM Synthesis**: Interprets scores in context, extracts qualitative insights from filing prose (risks, customers, suppliers, geographic exposure, MD&A), cross-references insider activity vs financial trends, assesses disclosure consistency
- Newer filings weighted more heavily than older filings in analysis
- Output: `CompanyAnalysis` per ticker (financial_health, insider_signal, red_flags, qualitative sections, cross-referenced insights, overall signal + confidence + thesis)
- US companies only (SEC EDGAR coverage)

**TechnicalAnalyst**: Deferred to after Phase 4 — Massive.com rate limits (5/min) create bottleneck, and yfinance covers most needs. Revisit once core pipeline is running.

**New files**: `processing/intelligence/models.py`, `processing/intelligence/specialists/us_company/`
**Extended**: `providers/yfinance/client.py`, `providers/yfinance/models.py`, `config_cache.py`

---

### Phase 3A: LangGraph Core + Two-Tier Layer 1
**Goal**: Build the LangGraph graph skeleton with two-tier Layer 1: signal discovery (Social + News) → ticker extraction → targeted deep analysis (CompanyAnalyst). Basic compiler assembles output.

**Reference implementations**: rgoerwit/ai-investment-agent (tiered sync, parallel fan-out), virattt/ai-hedge-fund (merge_dicts reducer), TauricResearch/TradingAgents (debate patterns). See `.claude/skills/langgraph-developing/` for project-specific patterns.

**LangGraph State Machine (Phase 3A scope)**:
```
START → Tier 1 (parallel, signal discovery):
  SocialSentimentAnalyst | NewsAnalyst
    → fan-in → extract_tickers (deterministic)
      → Tier 2 (targeted, deep analysis):
        CompanyAnalyst (only for extracted tickers)
          → Brief Compiler → END
```

#### Completed:

**CompanyAnalyst** [DONE]:
- Three-phase pipeline: yfinance fundamentals + quarterly financials → deterministic scoring (Piotroski, insider clusters, red flags) → LLM synthesis from 10-K/10-Q prose
- Uses yfinance quarterly data (same-day updates) instead of XBRL (delayed)
- Lives in `processing/intelligence/specialists/company/`

**raw_messages enrichment** [DONE]:
- `raw_messages` table has `impact_score` (SMALLINT) + `tickers` (TEXT[]) columns
- Flow 1 Stage 1 writes scores in single INSERT (no separate UPDATE)
- `db.get_raw_messages(since_hours, min_impact_score)` ready for NewsAnalyst

**YFinance quarterly financials** [DONE]:
- `get_quarterly_financials()` → `QuarterlyFinancials` model (income, balance sheet, cash flow)
- 5-6 quarters of data, Redis-cached

**SocialSentimentAnalyst** [DONE]:
- PydanticAI agent (OpenAI vsmart) reading `raw_tweets` table (last 24h)
- Tweets formatted by account with bias/credibility profiles from `x_accounts.py` (24 curated accounts)
- Tools: `verify_ticker` (us_tickers.json → yfinance Search fallback for non-US), `web_search` (5 calls max), `web_read`
- Extracts ticker mentions with context + macro themes (no scoring — information only)
- Output: `SocialSentimentAnalysis` (ticker_mentions, macro_themes, summary)
- Lives in `processing/intelligence/specialists/social_sentiment/`

**verify_ticker updated** [DONE]:
- Fallback changed from SearXNG to yfinance `Search` API
- Handles non-US tickers (D05.SI for DBS Singapore, 0700.HK for Tencent, etc.)

**New models** [DONE]: `SocialSentimentAnalysis`, `TickerMention`, `MacroTheme`

**NewsAnalyst** [DONE]:
- PydanticAI agent (OpenAI vsmart) reading enriched `raw_messages` (last 24h, impact_score >= 20)
- Groups related messages into `NewsStoryCluster` objects (same event = 1 cluster, not N separate mentions)
- Each cluster classified by `NewsEventType` (earnings, m&a, regulatory, macro, geopolitical, management, legal, product, financing, other)
- Per-ticker: `TickerMention` with context (shared model, no scoring)
- Tools: `verify_ticker` (us_tickers.json → yfinance fallback), `web_search` (5 max), `web_read`
- Output: `NewsAnalysis` (story_clusters, macro_themes, summary)
- Lives in `processing/intelligence/specialists/news/`

**New models** [DONE]: `NewsEventType`, `NewsStoryCluster`, `NewsAnalysis`

**SearXNG removed from ticker verification** [DONE]: Replaced with yfinance Search API fallback. Dead code (`search_ticker_analysis`, `_search_searxng`) removed from `web_search.py`.

**LangGraph infrastructure** [DONE]:
- `intelligence/state.py` — `IntelligenceState` TypedDict with `Annotated[list, add]` reducer for parallel company analyses
- `intelligence/graph.py` — `build_intelligence_graph(db, sec_edgar, yfinance, crawler)` closure factory. Topology: START → [social | news] (parallel) → extract_tickers → Send per ticker to CompanyAnalyst → compiler (defer=True) → END
- `intelligence/compiler.py` — deterministic brief assembly (no LLM). Filters errored analyses, merges macro themes from social + news.
- `langgraph>=1.1.6` dependency added
- Uses `Send` API for dynamic per-ticker fan-out, `defer=True` on compiler to wait for all parallel workers
- Unit tests: graph compilation, compiler logic, ticker extraction (9 tests)

#### Phase 3A Complete.

---

### Phase 3B: PriceAnalyst + MacroStrategist [DONE]
**Goal**: Add PriceAnalyst (Layer 2) and MacroStrategist (market regime assessment).

**Design principle** (from TradingAgents): Analysts are **information gatherers** — they extract, summarize, and structure key facts. They do NOT assign sentiment scores or make trading decisions. Scoring and decisions are deferred to the Trader (Phase 3D).

**LangGraph additions**:
```
Layer 2 (parallel per ticker via Send):
  CompanyAnalyst | PriceAnalyst
    + MacroStrategist (parallel with Layer 2, only needs FRED + Layer 1 themes)
```

#### Completed:

**Analyst refactor** [DONE]:
- Removed `sentiment_score` from all analyst outputs: `TickerMention`, `MacroTheme`, `InsiderSignal`, `CompanyAnalysis`
- Analysts focus on extracting valuable context — no scoring, no meta-commentary
- `sentiment_score` kept only on `MacroView`, `SectorTilt` (regime direction is inherently directional)

**MacroStrategist** [DONE]:
- PydanticAI agent (OpenAI vsmart) assessing market regime from FRED data + Layer 1 themes
- FRED series: VIX, 10Y/2Y yields, fed funds, unemployment — 10-observation trend history
- Tools: web_search (2 max), web_read (unlimited), get_fred_data
- Configurable via `MACRO_STRATEGIST_ENABLED` in .env
- Output: `MacroView` (regime, sentiment_score, key_drivers, sector_tilts, risks)
- Independent of Layer 2 per-ticker analysts — only needs FRED + Layer 1 themes

**EquityStrategist** [DONE — replaced by BullResearcher/BearResearcher in Phase 3C]:
- Was: sole decision maker producing ranked `TradeIdea` list
- Code removed in Phase 3C; context formatters extracted to `intelligence/context.py`

**PriceAnalyst** [DONE]:
- Three-phase pipeline per ticker: data gathering → pandas-ta indicators + options metrics → LLM interpretation
- **yfinance** (free, unlimited): 3mo OHLCV bars, quote, options snapshot (realized vol)
- **Massive** (up to 3 calls per ticker): contract lookup → ATM call EOD bars → ATM put EOD bars
- **IV self-computed** from Massive EOD close prices via Newton-Raphson BS inversion (verified: ~27% for AAPL ATM)
- **pandas-ta** (local from bars): RSI-14, MACD, EMA-8/21, ADX-14, ATR-14, BBands, OBV, pivot points, z-score
- Options metrics: ATM IV + skew (put IV / call IV), IV-RV spread, put/call volume ratio
- Notable setup detection: BB squeeze, RSI extremes, OBV divergence, IV-RV spread, elevated put skew
- Short interest available via CompanyAnalyst (yfinance `short_percent_of_float`)
- Runs **parallel per ticker** via Send (same as CompanyAnalyst) — Massive rate limiter handles queuing internally
- Output: `PriceAnalysis` (indicators + options metrics + notable setups + LLM narratives, no scoring)
- New dependency: `pandas-ta>=0.4`
- Lives in `processing/intelligence/specialists/price/`
- 26 unit tests (indicators, patterns, IV computation) + 6 integration tests

**Models** [DONE]: `MacroView`, `SectorTilt`

#### Phase 3B Complete.

---

### Phase 3C: Bull/Bear Debate (Layer 2)
**Goal**: Replace EquityStrategist with a TradingAgents-inspired bull/bear debate. Every extracted ticker gets debated by opposing researchers who receive ALL upstream context. No scoring — a future Trader node (Phase 3D) will make final decisions.

**Reference**: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) — bull/bear researchers debate with full analyst context, then a Research Manager (our future Trader) synthesizes.

**Design principle**: Analysts gather facts (Layer 1 + 2). Researchers argue opposing cases using those facts across all tickers in a single call. Neither side scores — they present evidence-based arguments. The Trader (Phase 3D) will be the sole decision maker.

**LangGraph topology**:
```
START → Layer 1 (parallel): SocialSentimentAnalyst | NewsAnalyst
  → extract_tickers
    → route_to_L2 (conditional):
        Always: MacroStrategist
        Per ticker: CompanyAnalyst | PriceAnalyst
      → l2_join (defer=True, waits for all)
        → l2_router (conditional):
            Has tickers → BullResearcher | BearResearcher (parallel, all tickers in one call each)
            No tickers  → Compiler (directly, macro-only brief)
          → Compiler (defer=True) → END
```

**Key changes from Phase 3B**:
- MacroStrategist runs **parallel** with per-ticker analysts (only needs FRED + Layer 1 themes)
- EquityStrategist and conviction gate removed
- `l2_join` is a defer=True sync barrier; `l2_router` conditionally skips debate when no tickers

**New agents**:
- **BullResearcher** (OpenAI vsmart): Builds strongest evidence-based case FOR investing. Receives ALL upstream context (social, news, company, price, macro) and analyzes all tickers in one call. Tools: web_search (3 max), web_read (unlimited).
- **BearResearcher** (OpenAI vsmart): Builds strongest evidence-based case AGAINST. Same context and tools.

**New models**: `DebateArgument` (ticker, argument, key_evidence), `DebateAnalysis` (role, arguments list, analysis_date)

**State fields**: `bull_analysis` + `bear_analysis` (single dict each, one writer, no reducer)

**Compiler**: Groups bull + bear arguments by ticker. Includes `price_analyses` in output. Surfaces pipeline errors via `errors` field. No scoring, no conviction splitting.

**New files**: `intelligence/context.py` (shared formatters), `intelligence/debate/__init__.py`, `intelligence/debate/bull.py`, `intelligence/debate/bear.py`

**Removed**: `equity_strategist` node, `conviction_gate` node, `equity_ideas` state field, `strategists/equity.py`, `TradeIdea` + `EquityIdeas` models

---

### Phase 3D: Trader (Research Manager)
**Goal**: Add a Trader node that synthesizes debate output into actionable trade ideas. Equivalent to TradingAgents' Research Manager.

**Deferred** — build after Phase 3C is validated.

**Trader** (OpenAI vsmart):
- Reads all debate arguments (bull + bear per ticker) + macro regime
- Makes decisive stance per ticker: Buy, Sell, or Hold (avoids defaulting to Hold)
- Produces `TradeIdea` with `sentiment_score` — the ONLY scored output in the pipeline
- Output: `EquityIdeas` (ranked trade ideas)

---

### Phase 4: Production Integration
**Goal**: Replace existing scheduler jobs with the unified pipeline. Ship to Discord.

**Changes**:
- Replace 3 daily scheduler jobs (twitter, events, market brief) with 1 unified pipeline trigger
- Compile LangGraph at startup in `agent/__main__.py`, wire all provider clients via closure
- New Discord embed formatters for intelligence brief (header + debates + supporting context + watchlist)
- Debated ideas show bull/bear arguments in embeds
- API endpoint: `POST /api/v1/intelligence/trigger` for manual runs
- Retire Market Brief (subsumed by MacroStrategist + Debate + Compiler)
- Event fetch job stays separate (populates DB before pipeline runs)

---

## Output: Daily Intelligence Brief

Lands in Discord each morning (~10:30am ET):
1. **Macro Regime** — risk-on/off, confidence, key drivers, sector tilts
2. **Ticker Debates** — per ticker: bull argument, bear argument, key evidence from each side
3. **Supporting Context** — top insider signals, technical setups, risk radar
4. **Watchlist Updates** — tickers added/removed with reasons

*Phase 3D adds*: Trade Ideas (ranked by conviction) with sentiment_score, thesis, structure, catalyst, timeframe.

## Cost & Performance

| Component | Model | Est. Cost/Day |
|---|---|---|
| SocialSentimentAnalyst | OpenAI vsmart | ~$0.15 |
| NewsAnalyst | OpenAI vsmart | ~$0.10 |
| CompanyAnalyst (per ticker, ~3-5/day) | OpenAI vsmart | ~$0.30-1.00 |
| PriceAnalyst (per ticker, ~3-5/day) | OpenAI vsmart | ~$0.05-0.15 |
| MacroStrategist | OpenAI vsmart | ~$0.08 |
| BullResearcher (per ticker, ~3-5/day) | OpenAI vsmart | ~$0.10-0.30 |
| BearResearcher (per ticker, ~3-5/day) | OpenAI vsmart | ~$0.10-0.30 |
| **Total** | | **~$0.88-2.08** |

Wall clock: ~80-130s end-to-end.

## Specs & Skills

- Architecture spec: `docs/superpowers/specs/2026-04-05-multi-agent-architecture-design.md`
- Phase 1 detail: `docs/twitterplan.md`
- LangGraph skill: `.claude/skills/langgraph-developing/`
