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

### Phase 2: USCompanyAnalyst + YFinance Extension
**Goal**: Build a comprehensive US company analysis agent combining SEC EDGAR + yfinance data, using a deterministic-scoring-then-LLM-synthesis pattern proven by top implementations (virattt/ai-hedge-fund, MarketSenseAI 2.0).

**YFinance Client Extension**:
- New `get_fundamentals(ticker)` method on `YFinanceClient` → `CompanyFundamentals` model
- Exposes pre-computed ratios from yfinance `.info`: current_ratio, quick_ratio, debt_to_equity, roe, roa, gross_margin, operating_margin, profit_margin, revenue_growth, free_cash_flow, ebitda, total_cash, total_debt, beta, short_interest, price_to_book, ev_to_ebitda, forward_eps, analyst_targets, sector, industry, business_summary, employees
- Redis-cached with configurable TTL

**USCompanyAnalyst** (OpenAI 5.2, 400k context):
- Three-phase pipeline per ticker:
  1. **Data Gathering** (no LLM): yfinance fundamentals + SEC EDGAR XBRL (8 quarters), insider transactions (Form 4, MSPR, Form 144), latest 10-K/10-Q filing prose via Crawl4AI
  2. **Deterministic Scoring** (no LLM): Piotroski F-Score (0-9), Beneish M-Score (earnings manipulation), insider cluster detection, red flag detection
  3. **LLM Synthesis**: Interprets scores in context, extracts qualitative insights from filing prose (risks, customers, suppliers, geographic exposure, MD&A), cross-references insider activity vs financial trends, assesses disclosure consistency
- Newer filings weighted more heavily than older filings in analysis
- Output: `CompanyAnalysis` per ticker (financial_health, insider_signal, red_flags, qualitative sections, cross-referenced insights, overall signal + confidence + thesis)
- US companies only (SEC EDGAR coverage)

**TechnicalAnalyst**: Deferred to after Phase 4 — Massive.com rate limits (5/min) create bottleneck, and yfinance + XBRL cover most needs. Revisit once core pipeline is running.

**New files**: `processing/intelligence/models.py`, `processing/intelligence/specialists/us_company.py`, `processing/intelligence/specialists/scoring.py`
**Extended**: `providers/yfinance/client.py`, `providers/yfinance/models.py`, `config_cache.py`

---

### Phase 3: LangGraph Pipeline
**Goal**: Wire all agents into a LangGraph StateGraph with multi-round debate.

**LangGraph State Machine**:
```
Layer 1 (parallel fan-out):
  SocialSentimentAnalyst | EventCalendarAnalyst | USCompanyAnalyst

Layer 2 (fan-in, sequential):
  MacroStrategist → EquityStrategist

Gate: conviction >= 0.7 AND 2+ source convergence

Layer 2.5 (debate loop, 2 rounds):
  BullAdvocate ↔ BearAdvocate → Adjudicator

Layer 3 (deterministic):
  Brief Compiler → DailyIntelligenceBrief
```

**New agents**:
- **MacroStrategist** (Sonnet): Regime assessment (risk-on/off), sector tilts, macro trade ideas. Tools: FRED observations, web search
- **EquityStrategist** (Sonnet): Cross-signal convergence synthesis. Pre-computed convergence map shows which tickers appear in 2+ Layer 1 reports. Tools: quotes, options snapshots, web search
- **BullAdvocate** (Haiku): Argues strongest case for each gated trade idea. No tools, pure argumentation
- **BearAdvocate** (Haiku): Attacks the thesis, finds contradictions. No tools
- **Adjudicator** (Sonnet): Weighs both sides, adjusts conviction, produces final recommendation. 1 web search max
- **Brief Compiler** (no LLM): Ranks ideas by conviction, assembles `DailyIntelligenceBrief`

**State**: `IntelligenceState` TypedDict with `Annotated` reducers for safe concurrent writes. Deps passed via closure factory.

**New files**: `intelligence/state.py`, `intelligence/graph.py`, `strategists/macro.py`, `strategists/equity.py`, `debate/bull.py`, `debate/bear.py`, `debate/adjudicator.py`, `intelligence/compiler.py`

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
| SocialSentimentAnalyst | Sonnet (vsmart) | ~$0.15 |
| EventCalendarAnalyst | Sonnet x3 + vsmart | ~$0.15 |
| USCompanyAnalyst | OpenAI 5.2 | ~$0.10-0.30 |
| MacroStrategist | Sonnet | ~$0.08 |
| EquityStrategist | Sonnet | ~$0.12 |
| Debate (avg 1.5 ideas/day, 2 rounds) | Haiku x4 + Sonnet | ~$0.20 |
| **Total** | | **~$0.70-1.20** |

Wall clock: ~80-130s end-to-end.

## Specs & Skills

- Architecture spec: `docs/superpowers/specs/2026-04-05-multi-agent-architecture-design.md`
- Phase 1 detail: `docs/twitterplan.md`
- LangGraph skill: `.claude/skills/langgraph-developing/`
