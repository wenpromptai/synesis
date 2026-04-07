# Intelligence Pipeline Architecture

LangGraph state-machine that transforms raw social/news signals into structured per-ticker bull/bear briefs with configurable multi-round debate.

Source: `src/synesis/processing/intelligence/`

## Graph Topology

```
                         START
                           |
                    +--------------+
                    |              |
            social_sentiment  news_analyst           ← Layer 1 (parallel)
                    |              |
                    +--------------+
                           |
                    extract_tickers                  ← Deterministic
                           |
                    route_to_L2()                    ← Conditional: Send per ticker
                           |
              +------------+------------+
              |            |            |
       company_analyst  price_analyst  macro_strategist  ← Layer 2 (parallel fan-out)
        (per ticker)    (per ticker)     (singleton)
              |            |            |
              +------------+------------+
                           |
                       l2_join                        ← Sync barrier (defer=True)
                           |
                      l2_router()                     ← Conditional: configurable by debate_rounds
                     /        |        \
          [rounds=0]    [rounds>=1]   [no tickers]
              |              |              |
       +-----------+    ticker_debate       |
       |           |    (per ticker)        |         ← Layer 3: Debate
  bull_researcher  |     subgraph:          |
   bear_researcher |  bull⇄bear loop        |
    (per ticker,   |    N rounds            |
     parallel)     |         |              |
       +-----------+         |              |
              \              |             /
                      compiler                       ← Deterministic assembly (defer=True)
                           |
                          END
```

Ref: `graph.py:298-344` — graph construction + edge wiring

## State (`IntelligenceState`)

Defined in `state.py:13-42`.

| Field              | Type         | Reducer        | Writer                                   |
| ------------------ | ------------ | -------------- | ---------------------------------------- |
| `current_date`     | `str`        | overwrite      | input                                    |
| `social_analysis`  | `dict`       | overwrite      | social_sentiment                         |
| `news_analysis`    | `dict`       | overwrite      | news_analyst                             |
| `target_tickers`   | `list[str]`  | overwrite      | extract_tickers                          |
| `company_analyses` | `list[dict]` | `add` (append) | company_analyst × N                      |
| `price_analyses`   | `list[dict]` | `add` (append) | price_analyst × N                        |
| `macro_view`       | `dict`       | overwrite      | macro_strategist                         |
| `bull_analyses`    | `list[dict]` | `add` (append) | bull_researcher × N or ticker_debate × N |
| `bear_analyses`    | `list[dict]` | `add` (append) | bear_researcher × N or ticker_debate × N |
| `brief`            | `dict`       | overwrite      | compiler                                 |

All `add` reducer fields accumulate results from parallel Send fan-out nodes (one per ticker). When using multi-round debate (`rounds>=1`), each ticker contributes multiple bull/bear entries (one per round) — compiler takes the last round per ticker.

## Layer 1: Signal Discovery (parallel)

Two PydanticAI agents run in parallel, scanning the last 24h of ingested data.

### SocialSentimentAnalyst

Source: `specialists/social_sentiment/agent.py`

- **Input**: Raw tweets from DB, grouped by account with profile context from `x_accounts.py` (bias, focus areas)
- **Tools**: `verify_ticker`, `web_search` (5 calls), `web_read`
- **Output**: `SocialSentimentAnalysis` — `ticker_mentions[]`, `macro_themes[]`, `summary`
- **Model**: `models.py:141-148`

### NewsAnalyst

Source: `specialists/news/agent.py`

- **Input**: Pre-scored `raw_messages` from DB (`impact_score >= 20`)
- **Tools**: `verify_ticker`, `web_search` (5 calls), `web_read`
- **Output**: `NewsAnalysis` — `story_clusters[]` (event_type, urgency, key_facts, tickers), `macro_themes[]`, `summary`
- **Models**: `NewsStoryCluster` (`models.py:170-178`), `NewsAnalysis` (`models.py:181-188`), `NewsEventType` enum (`models.py:155-167`)

## Ticker Extraction (deterministic)

Source: `graph.py:102-119`

Collects all unique tickers from both Layer 1 outputs (social `ticker_mentions` + news `story_clusters`), deduplicates into sorted `target_tickers[]`.

## Layer 2: Multi-Agent Analysis (parallel fan-out)

`route_to_L2()` (`graph.py:123-139`) conditional edge uses `Send` to fan out:

- One `macro_strategist` invocation (always, singleton)
- One `company_analyst` + one `price_analyst` per ticker via `Send("node", {**state, "ticker": t})`

All run in parallel.

### CompanyAnalyst (per ticker)

Source: `specialists/company/agent.py`, scoring in `specialists/company/scoring.py`

Three-phase pipeline — only Phase 3 uses LLM:

| Phase                  | What                                                                                                                                                                                                   | Source                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| 1. Data Gather         | `asyncio.gather`: yfinance fundamentals + quarterly financials, SEC EDGAR insider txns + sentiment + Form 144 + late filing alerts, 10-K/10-Q text                                                     | yfinance, SEC EDGAR, Crawl4AI |
| 2. Deterministic Score | Piotroski F-Score (9 signals from quarterly data), Beneish M-Score, InsiderSignal (MSPR, buy/sell counts, cluster detection, C-suite activity), RedFlag detection (late filings, cash flow divergence) | Phase 1 data                  |
| 3. LLM Synthesis       | business_summary, earnings_quality, risk_assessment, geographic_exposure, key_customers_suppliers, insider_vs_financials, disclosure_consistency, primary_thesis, key_risks, monitoring_triggers       | Phase 1+2 as prompt context   |

**Output**: `CompanyAnalysis` (`models.py:104-133`) — merges Phase 2 deterministic fields + Phase 3 LLM fields.

### PriceAnalyst (per ticker)

Source: `specialists/price/agent.py`, indicators in `specialists/price/indicators.py`

Three-phase pipeline — only Phase 3 uses LLM:

| Phase                    | What                                                                                                                                                                                                                                                     | Source                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 1. Data Gather           | yfinance quote + 3mo daily bars + options snapshot; Massive ATM options contracts (nearest monthly DTE > 20)                                                                                                                                             | yfinance, Massive           |
| 2. Deterministic Compute | Technical: EMA 8/21, ADX, RSI 14, MACD, ATR%, Bollinger Bands, z-score, volume ratio, OBV, support/resistance. Options: ATM IV via Black-Scholes inversion (Newton-Raphson on Massive EOD), realized vol 30d, IV/RV spread, put/call vol ratio, ATM skew | pandas-ta, custom BS solver |
| 3. LLM Synthesis         | technical_narrative (3-5 sentences), options_narrative (3-5 sentences)                                                                                                                                                                                   | Phase 1+2 as prompt context |

**Output**: `PriceAnalysis` (`models.py:246-284`) — all deterministic indicators + LLM narratives + notable_setups[].

### MacroStrategist (singleton)

Source: `strategists/macro.py`

- **Input**: FRED data (`_FRED_SERIES`: VIXCLS, DGS10, DGS2, FEDFUNDS, UNRATE — 10-observation history each, plus computed yield curve spread) + aggregated Layer 1 macro themes
- **Tools**: `web_search` (2 calls), `web_read`, `get_fred_data`
- **Output**: `MacroView` (`models.py:209-222`) — regime (risk_on/risk_off/transitioning/uncertain), sentiment_score (-1 to 1), key_drivers[], sector_tilts[], risks[]
- **Toggle**: Skipped if `settings.macro_strategist_enabled=False` or FRED client unavailable (`graph.py:186-188`)

## Layer 2 Join + Routing

### `l2_join` (sync barrier)

Source: `graph.py:199-201`

No-op node with `defer=True` (`graph.py:309`). All Layer 2 nodes (`company_analyst`, `price_analyst`, `macro_strategist`) edge into it (`graph.py:327-329`). The `defer` flag makes it wait for all upstream Send fan-out nodes to complete before proceeding.

### `l2_router` (conditional)

Source: `graph.py:203-222`

Routes from `l2_join` based on ticker availability and `settings.debate_rounds`:

- **No tickers** → `["compiler"]` (skip debate entirely)
- **`debate_rounds=0`** → `Send("bull_researcher", ...)` + `Send("bear_researcher", ...)` per ticker (parallel, no debate)
- **`debate_rounds>=1`** → `Send("ticker_debate", ...)` per ticker (multi-round debate subgraph)

Wired via `add_conditional_edges` (`graph.py:332-336`).

## Layer 3: Debate

Configurable via `settings.debate_rounds` (default: 0).

### Mode A: Parallel independent (`debate_rounds=0`)

Each researcher is invoked **once per ticker** via Send. No debate history — bull and bear work independently.

### Mode B: Multi-round debate subgraph (`debate_rounds>=1`)

Source: `debate/subgraph.py`

A compiled LangGraph **subgraph** is invoked per ticker from the `ticker_debate` node (`graph.py:254-284`). The subgraph runs a cyclic bull⇄bear loop:

```
START → bull_debate → bear_debate → _should_continue
                                       ├─ round < max_rounds → bull_debate  (loop)
                                       └─ round >= max_rounds → END
```

#### Debate Subgraph State (`DebateState`)

Defined in `debate/subgraph.py:26-40`.

| Field              | Type         | Reducer        | Purpose                                           |
| ------------------ | ------------ | -------------- | ------------------------------------------------- |
| `ticker`           | `str`        | —              | Target ticker (from Send)                         |
| `current_date`     | `str`        | —              | Pipeline date                                     |
| `social_analysis`  | `dict`       | —              | Layer 1 context (read-only)                       |
| `news_analysis`    | `dict`       | —              | Layer 1 context (read-only)                       |
| `company_analyses` | `list[dict]` | —              | Layer 2 context (read-only)                       |
| `price_analyses`   | `list[dict]` | —              | Layer 2 context (read-only)                       |
| `debate_history`   | `list[dict]` | `add` (append) | Accumulates all bull/bear arguments across rounds |
| `round`            | `int`        | overwrite      | Current round (incremented after each bear turn)  |
| `max_rounds`       | `int`        | —              | Configured debate_rounds value                    |

Each round: bull argues (seeing all prior history) → bear argues (seeing all prior history including bull's latest) → round increments. The `_should_continue` function loops back to `bull_debate` if `round < max_rounds`, otherwise ends.

The `ticker_debate` node splits the final `debate_history` into `bull_analyses` + `bear_analyses` for the main graph state (`graph.py:276-278`).

### Shared: BullResearcher and BearResearcher

Both agents accept an optional `debate_history` parameter (`debate/bull.py:132`, `debate/bear.py:133`):

- **Without history** (rounds=0): standard per-ticker analysis, no debate instructions in system prompt
- **With history** (rounds>=1): prior arguments formatted via `format_debate_history()` (`context.py:286-300`) and appended to prompt. System prompt includes debate-specific instructions requiring direct counter-arguments

Per-ticker context formatters (`context.py:308-379`):

- `format_social_context_for_ticker(state, ticker)`
- `format_news_context_for_ticker(state, ticker)`
- `format_company_context_for_ticker(state, ticker)`
- `format_price_context_for_ticker(state, ticker)`

Macro context is **not** passed to debate agents — deferred to future Trader node.

#### BullResearcher

Source: `debate/bull.py`

- **Tools**: `web_search` (3 calls — `_WEB_SEARCH_CAP`), `web_read`
- **Output**: `TickerDebate(role="bull")` (`models.py:230-238`) — argument, key_evidence, round number

#### BearResearcher

Source: `debate/bear.py`

- **Tools**: `web_search` (3 calls — `_WEB_SEARCH_CAP`), `web_read`
- **Output**: `TickerDebate(role="bear")` — argument, key_evidence, round number

## Compilation (deterministic, defer=True)

Source: `compiler.py`

No LLM. `defer=True` (`graph.py:313`) waits for all debate nodes. Groups bull + bear by ticker, **sorted by round** so last round wins (`compiler.py:35, 40`). Assembles final brief:

```python
{
    "date": "2026-04-06",
    "macro": { "regime", "sentiment_score", "key_drivers", "sector_tilts", "risks" },
    "debates": [
        { "ticker": "NVDA", "bull": {...}, "bear": {...} },  # last-round arguments
        ...
    ],
    "l1_summary": { "social": "...", "news": "..." },
    "tickers_analyzed": [...],
    "company_analyses": [...],   # valid only (errors filtered)
    "price_analyses": [...],     # valid only (errors filtered)
    "macro_themes": [...],       # merged social + news themes
    "ticker_mentions": { "social": [...], "news_clusters": [...] },
    "messages_analyzed": count,
    "errors": {                  # per-node failure tracking
        "social_failed": bool,
        "news_failed": bool,
        "company_failures": [ticker, ...],
        "price_failures": [ticker, ...],
        "bull_failures": [ticker, ...],
        "bear_failures": [ticker, ...],
        "macro_failed": bool,
    },
}
```

## Agent Summary

| Agent                    | Model Tier | Tools                               | Web Budget  | Output Type               |
| ------------------------ | ---------- | ----------------------------------- | ----------- | ------------------------- |
| SocialSentimentAnalyst   | vsmart     | verify_ticker, web_search, web_read | 5           | `SocialSentimentAnalysis` |
| NewsAnalyst              | vsmart     | verify_ticker, web_search, web_read | 5           | `NewsAnalysis`            |
| CompanyAnalyst (Phase 3) | vsmart     | none (deterministic Phase 1-2)      | 0           | `CompanyAnalysis`         |
| PriceAnalyst (Phase 3)   | vsmart     | none (deterministic Phase 1-2)      | 0           | `PriceAnalysis`           |
| MacroStrategist          | vsmart     | web_search, web_read, get_fred_data | 2           | `MacroView`               |
| BullResearcher           | vsmart     | web_search, web_read                | 3 per round | `TickerDebate`            |
| BearResearcher           | vsmart     | web_search, web_read                | 3 per round | `TickerDebate`            |

## Key Design Decisions

1. **No scoring/trading at agent level** — agents provide analysis, not BUY/SELL signals. Trading decisions deferred to a future Trader agent downstream (`models.py:1-9`).

2. **Configurable debate depth** — `debate_rounds=0` gives fast parallel bull/bear; `debate_rounds>=1` enables adversarial multi-round debate where each side must directly counter the other's arguments. The subgraph loop (`debate/subgraph.py:60-64`) controls the cycle.

3. **Debate history accumulation** — `DebateState.debate_history` uses `add` reducer to accumulate all arguments across rounds. Both researchers see the full history each turn, enabling increasingly targeted counter-arguments (`context.py:286-300`).

4. **Last-round-wins compilation** — compiler sorts by round and takes the final argument per ticker per side (`compiler.py:35, 40`), so the most refined version of each case goes into the brief.

5. **Explicit sync barrier** — `l2_join` node with `defer=True` (`graph.py:309`) cleanly separates Layer 2 fan-out from debate. The `l2_router` conditional then decides debate mode based on `debate_rounds` setting (`graph.py:203-222`).

6. **Web search budgets** — hard caps per agent per round prevent runaway API costs (Layer 1: 5 each, Debate: 3 per round per side, Macro: 2).

7. **Graceful degradation** — EDGAR failures don't block CompanyAnalyst; Massive failures don't block PriceAnalyst; FRED failures don't block MacroStrategist. Layer 1 failures return `{"error": True}` for downstream visibility (`graph.py:87, 98`).

8. **Closure capture** — provider clients (DB, yfinance, SEC EDGAR, FRED, Massive, Crawl4AI) captured at graph build time via `build_intelligence_graph()` closure, never serialized in state (`graph.py:62-69`).

9. **Send semantics for Layer 2 + Layer 3** — `Send("node", {**state, "ticker": t})` creates independent state copies per ticker. Results merge back via `Annotated[list, add]` reducers (`state.py:31-32, 38-39`).

10. **Per-node error tracking** — compiler surfaces failures per ticker (`bull_failures`, `bear_failures`, `company_failures`, `price_failures`) and per Layer 1 node (`social_failed`, `news_failed`) for downstream visibility (`compiler.py:83-97`).
