# Trader Agent — Phase 3D Design

**Date:** 2026-04-07
**Status:** Approved
**Scope:** Add a Trader node to the intelligence pipeline that synthesizes debate output into actionable trade ideas.

## Problem

The pipeline currently ends with raw bull/bear debate arguments per ticker. Nobody makes a decision. A Trader agent reads the debate + all supporting context and produces scored, actionable `TradeIdea` outputs — the only scored output in the pipeline.

## Configuration

Two modes via `TRADER_MODE` env var:

- **`per_ticker`** (default): Trader receives one ticker at a time via `Send()` fan-out. Gets macro context + that ticker's debate + company/price data.
- **`portfolio`**: Trader receives all tickers in one call. Can weigh cross-ticker correlation, concentration, and capital allocation. Produces a `portfolio_note` with cross-ticker observations.

Both modes receive macro context.

## Graph Topology

```
(existing) → l2_join → l2_router → debate nodes
  → trader_gate (defer=True, waits for all debate output)
    → trader_router (conditional):
        Has tickers + per_ticker mode → Send per ticker to trader node
        Has tickers + portfolio mode  → single Send to trader node
        No tickers                    → compiler directly
      → compiler (defer=True) → END
```

### Why `trader_gate` instead of reusing `compiler`

The Trader needs ALL debate results before it can run — it must see every ticker's bull/bear arguments. Without `trader_gate`, the `defer=True` on compiler would wait for all work, but the Trader would start receiving Sends as soon as the first debate completes. The `trader_gate` (defer=True) ensures the Trader only starts after all debate nodes have finished.

This matches the existing `l2_join` pattern: a defer barrier before a conditional router.

### New nodes

| Node | Type | Purpose |
|------|------|---------|
| `trader_gate` | Sync barrier (defer=True) | Waits for all debate output before routing to Trader |
| `trader` | LLM agent node (via Send) | Makes trade decisions |

`trader_router` is a conditional edge function, not a node.

### Edge wiring

```python
# Debate outputs feed into trader_gate (defer waits for all)
graph.add_edge("bull_researcher", "trader_gate")
graph.add_edge("bear_researcher", "trader_gate")
graph.add_edge("ticker_debate", "trader_gate")

# Conditional routing from trader_gate
graph.add_conditional_edges(
    "trader_gate",
    trader_router,
    ["trader", "compiler"],
)

# Trader feeds into compiler (defer waits for all trader Sends)
graph.add_edge("trader", "compiler")
```

`l2_router` keeps `"compiler"` in its path map for the no-tickers case. When there are no tickers, `l2_router` returns `["compiler"]` and the debate + trader layers are skipped entirely.

## Models

All in `processing/intelligence/models.py`.

### TradeIdea

```python
class TradeIdea(BaseModel):
    """A trade recommendation from the Trader — the ONLY scored output."""

    ticker: str = Field(min_length=1)
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Direction + conviction. -1.0 (strong sell) to 1.0 (strong buy). 0 = skip.",
    )
    thesis: str = ""
    trade_structure: str = ""
    catalyst: str = ""
    timeframe: str = ""
    key_risk: str = ""
    analysis_date: date
```

Sentiment score calibration:
- +-0.8 to +-1.0: high conviction
- +-0.4 to +-0.7: moderate
- +-0.1 to +-0.3: weak
- 0: skip

### TraderOutput

```python
class TraderOutput(BaseModel):
    """Full Trader output — wraps one or more TradeIdeas."""

    trade_ideas: list[TradeIdea] = Field(default_factory=list)
    skipped_tickers: list[str] = Field(default_factory=list)
    portfolio_note: str = ""
    analysis_date: date
```

- `skipped_tickers`: tickers the Trader chose to pass on (lopsided debate, insufficient data, etc.)
- `portfolio_note`: cross-ticker observations, only populated in portfolio mode

## State Changes

### state.py

Add one new field:

```python
# Trader output (multiple parallel writers in per_ticker mode, needs reducer)
trade_ideas: Annotated[list[dict[str, Any]], add]
```

## Trader Agent

### File structure

```
processing/intelligence/trader/
├── __init__.py
└── trader.py
```

### Agent details

| Property | Value |
|----------|-------|
| Model | vsmart (gpt-5.2) |
| Tools | web_search (3 max), web_read (unlimited) |
| Deps | `TraderDeps(current_date, web_search_calls)` |
| Output type | `TraderOutput` |

### System prompt

Tells the Trader:
- You are the sole decision maker in the pipeline
- You receive macro regime + debate arguments (bull/bear) + company/price data
- Make a decisive call. Avoid defaulting to hold — if not convinced, skip the ticker
- Suggest a concrete trade structure (shares, specific options spread, etc.)
- In portfolio mode: consider cross-ticker correlation and concentration

### Per-ticker mode input

Built using existing context formatters in `context.py` plus a new `format_macro_context` (already exists, currently unused — wired in for Trader):

```
## Ticker: NVDA
[macro context]
[bull argument + key evidence]
[bear argument + key evidence]
[company analysis summary]
[price analysis summary]
```

### Portfolio mode input

All tickers concatenated with macro context at the top:

```
## Macro Regime
[macro context]

## Ticker: NVDA
[bull/bear + company + price]

## Ticker: AAPL
[bull/bear + company + price]
...
```

### Error handling

Same pattern as all other nodes:

```python
async def trader_node(state: dict[str, Any]) -> dict[str, Any]:
    ticker = state["ticker"]
    try:
        result = await analyze_trade(state, current_date)
        return {"trade_ideas": [idea.model_dump(mode="json") for idea in result.trade_ideas]}
    except Exception:
        logger.exception("Trader failed", ticker=ticker)
        return {"trade_ideas": [{"ticker": ticker, "error": True}]}
```

## Compiler Changes

### compiler.py

- Include `trade_ideas` in brief output, ranked by `abs(sentiment_score)` descending
- Add `trader_failures` to errors dict:
  ```python
  "trader_failures": [
      item["ticker"] for item in trade_ideas if item.get("error") and "ticker" in item
  ],
  ```
- Existing `debates` stay in the brief — debate is the reasoning, trade ideas are the decisions

## Config Changes

### config.py

```python
trader_mode: str = Field(
    default="per_ticker",
    description="Trader evaluation mode: 'per_ticker' or 'portfolio'",
)
```

## Context Formatter

Wire `format_macro_context` (already exists in `context.py`, built for Phase 3D) into the Trader's prompt builder. No new formatter needed — it was forward-engineered in Phase 3C.

## Testing

### Unit tests (test_intelligence_graph.py)

- Graph structure: `trader_gate`, `trader` nodes exist
- `trader_router`: correct per-ticker Sends for per_ticker mode
- `trader_router`: single Send for portfolio mode
- `trader_router`: skips to compiler when no tickers
- Compiler includes `trade_ideas` ranked by abs(sentiment_score)
- Compiler tracks `trader_failures` in errors
- Full graph execution with mocked Trader (both modes)
- Partial failure: one ticker fails, others succeed

### Model tests

- `TradeIdea.sentiment_score` bounded to [-1, 1]
- `TradeIdea.ticker` requires min_length=1
- `TraderOutput` validation

## Cost Estimate

With GPT-5.2 ($1.75/1M input, $14/1M output), 7 tickers/day:

| Mode | Input | Output | Daily cost |
|------|-------|--------|------------|
| per_ticker | 7 x ~6k = 42k | 7 x ~500 = 3.5k | ~$0.12 |
| portfolio | ~30k | ~3.5k | ~$0.10 |

Adds ~$0.10-0.12/day to existing pipeline cost.
