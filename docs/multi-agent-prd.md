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
- Extracts both ticker-specific signals (any ticker any account mentions) and macro themes (non-ticker trading ideas like risk-off, sector rotation)
- Output: `SocialSentimentAnalysis` (ticker_mentions: list[TickerMention], macro_themes: list[MacroTheme], summary)
- Lives in `processing/intelligence/specialists/social_sentiment/`

**verify_ticker updated** [DONE]:
- Fallback changed from SearXNG to yfinance `Search` API
- Handles non-US tickers (D05.SI for DBS Singapore, 0700.HK for Tencent, etc.)

**New models** [DONE]: `SocialSentimentAnalysis`, `TickerMention`, `MacroTheme`

**NewsAnalyst** [DONE]:
- PydanticAI agent (OpenAI vsmart) reading enriched `raw_messages` (last 24h, impact_score >= 20)
- Groups related messages into `NewsStoryCluster` objects (same event = 1 cluster, not N separate mentions)
- Each cluster classified by `NewsEventType` (earnings, m&a, regulatory, macro, geopolitical, management, legal, product, financing, other)
- Per-ticker: `NewsTickerMention` with sentiment + magnitude (low/medium/high) + confidence (0-1) + `is_direct_impact` (False for sector drag)
- Tools: `verify_ticker` (us_tickers.json → yfinance fallback), `web_search` (5 max), `web_read`
- Output: `NewsAnalysis` (story_clusters, macro_themes, summary)
- Lives in `processing/intelligence/specialists/news/`

**New models** [DONE]: `NewsEventType`, `NewsTickerMention`, `NewsStoryCluster`, `NewsAnalysis`

**SearXNG removed from ticker verification** [DONE]: Replaced with yfinance Search API fallback. Dead code (`search_ticker_analysis`, `_search_searxng`) removed from `web_search.py`.

#### Remaining:

**extract_tickers node** (deterministic, no LLM):
- Collects all tickers mentioned in Social + News outputs
- Deduplicates, produces `target_tickers: list[str]`

**CompanyAnalyst LangGraph node wrapper**:
- Wraps existing `analyze_company()` as a LangGraph node
- Runs only for `target_tickers` (not full watchlist)

**LangGraph infrastructure**:
- `intelligence/state.py` — `IntelligenceState` TypedDict with `Annotated` reducers
- `intelligence/graph.py` — `build_intelligence_graph()` closure factory, all nodes + edges
- `intelligence/compiler.py` — basic assembly of all outputs (no LLM). Expanded in Phase 3C.
- Add `langgraph>=1.1` dependency to `pyproject.toml`

**New model**: `NewsAnalysis`

---

### Phase 3B: Strategists (Layer 2)
**Goal**: Add MacroStrategist + EquityStrategist as Layer 2 nodes that synthesize Layer 1 outputs. Add conviction gate for debate routing.

**LangGraph additions**:
```
Layer 1 fan-in → MacroStrategist → EquityStrategist → Gate
  Gate: conviction >= 0.7 AND 2+ source convergence
    → debate (Phase 3C) OR compiler
```

**New agents**:
- **MacroStrategist** (OpenAI, smart): Regime assessment (risk-on/off), sector tilts, macro trade ideas. Tools: FRED observations, web search (2 calls max). Input: social macro themes + news signals + FRED data.
- **EquityStrategist** (OpenAI, smart): Cross-signal convergence synthesis. Pre-computes convergence map: `{ticker: [sources]}` showing which tickers appear in 2+ Layer 1 reports. Tools: quotes, options snapshots (3 calls max), web search (2 calls max). Output: `TradeIdea` list ranked by conviction.

**Gate logic**: Deterministic routing function. Ideas with conviction >= 0.7, 2+ source convergence, and direction in (long, short) route to debate. Others go directly to compiler.

**New files**: `intelligence/strategists/macro.py`, `intelligence/strategists/equity.py`
**Models**: `MacroView`, `SectorTilt`, `TradeIdea`, `EquityIdeas`

---

### Phase 3C: Debate Loop (Layer 2.5)
**Goal**: Add bull/bear debate with adjudicator for high-conviction trade ideas. Uses per-field state for parallel safety (from ai-investment-agent pattern).

**LangGraph additions**:
```
Gate → parallel per round:
  BullAdvocate R1 | BearAdvocate R1 → Debate Sync R1
    → BullAdvocate R2 | BearAdvocate R2 → Debate Sync R2
      → Adjudicator → Compiler
```

**New agents**:
- **BullAdvocate** (OpenAI, fast): Argues strongest case for each gated trade idea. No tools, pure argumentation. Sees full debate history via state reducer.
- **BearAdvocate** (OpenAI, fast): Attacks the thesis, finds contradictions. No tools. Increments debate round counter.
- **Adjudicator** (OpenAI, smart): Weighs both sides, adjusts conviction, produces final recommendation. 1 web search max for fact-checking.

**Debate state**: Per-field ownership pattern (not shared key) for parallel safety:
```python
class DebateState(TypedDict):
    bull_r1: str      # Bull owns
    bear_r1: str      # Bear owns
    bull_r2: str      # Bull owns
    bear_r2: str      # Bear owns
    round: int
```

**Output**: `AdjudicatedIdea` (revised conviction, verdict, bull/bear summaries, stop-loss trigger)

**Compiler expanded**: Ranks ideas by final conviction (adjudicated if debated, raw otherwise). Assembles full `DailyIntelligenceBrief`.

**New files**: `intelligence/debate/bull.py`, `intelligence/debate/bear.py`, `intelligence/debate/adjudicator.py`
**Models**: `AdjudicatedIdea`, `DailyIntelligenceBrief`

---

### Phase 4: Production Integration
**Goal**: Replace existing scheduler jobs with the unified pipeline. Ship to Discord.

**Changes**:
- Replace 3 daily scheduler jobs (twitter, events, market brief) with 1 unified pipeline trigger
- Compile LangGraph at startup in `agent/__main__.py`, wire all provider clients via closure
- New Discord embed formatters for intelligence brief (header + trade ideas + supporting context + watchlist)
- Debated ideas show bull/bear summary in embeds
- API endpoint: `POST /api/v1/intelligence/trigger` for manual runs
- Retire Market Brief (subsumed by MacroStrategist + EquityStrategist + Compiler)
- Event fetch job stays separate (populates DB before pipeline runs)

---

## Output: Daily Intelligence Brief

Lands in Discord each morning (~10:30am ET):
1. **Macro Regime** — risk-on/off, confidence, key drivers, sector tilts
2. **Trade Ideas** (ranked by conviction) — ticker, direction, thesis, structure, catalyst. Debated ideas include bull/bear summary + stop-loss trigger
3. **Quick Takes** — lower conviction / watch-only ideas, 1 line each
4. **Supporting Context** — top insider signals, technical setups, risk radar
5. **Watchlist Updates** — tickers added/removed with reasons

## Cost & Performance

| Component | Model | Est. Cost/Day |
|---|---|---|
| SocialSentimentAnalyst | OpenAI vsmart | ~$0.15 |
| NewsAnalyst | No LLM (reader) | ~$0.00 |
| CompanyAnalyst (per ticker, ~3-5/day) | OpenAI vsmart | ~$0.30-1.00 |
| MacroStrategist | Sonnet | ~$0.08 |
| EquityStrategist | Sonnet | ~$0.12 |
| Debate (avg 1.5 ideas/day, 2 rounds) | Haiku x4 + Sonnet | ~$0.20 |
| **Total** | | **~$0.70-1.20** |

Wall clock: ~80-130s end-to-end.

## Specs & Skills

- Architecture spec: `docs/superpowers/specs/2026-04-05-multi-agent-architecture-design.md`
- Phase 1 detail: `docs/twitterplan.md`
- LangGraph skill: `.claude/skills/langgraph-developing/`
