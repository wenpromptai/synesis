# Phase 1: Raw Tweet Storage + Remove yfinance Tools

## Context
The Twitter agent currently fetches live market data (`get_quote`, `get_options_snapshot` via yfinance) during analysis. This burns tool budget on data the LLM doesn't need to form opinions. Additionally, tweets aren't persisted — they're fetched and passed directly to the LLM, making them unavailable for the future multi-agent intelligence pipeline.

This phase makes two surgical changes:
1. **Add `raw_tweets` table** — persist each tweet as a row before LLM analysis
2. **Remove yfinance tools** — the LLM focuses on tweet extraction + web research only

Everything else (models, Discord output, watchlist logic) stays as-is. The full model rework (sentiment_score, SocialSentimentAnalyst) happens in Phase 3 of the multi-agent pipeline.

## Changes

### 1. Database schema — `database/init.sql`

**New `raw_tweets` table:**
```sql
CREATE TABLE raw_tweets (
    id BIGSERIAL PRIMARY KEY,
    account_username TEXT NOT NULL,
    tweet_text TEXT NOT NULL,
    tweet_timestamp TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tweet_url TEXT,
    UNIQUE (account_username, tweet_timestamp, md5(tweet_text))
);

CREATE INDEX idx_raw_tweets_fetched ON raw_tweets (fetched_at DESC);
CREATE INDEX idx_raw_tweets_account ON raw_tweets (account_username, tweet_timestamp DESC);
```

Apply live via: `docker exec` the ALTER TABLE on the running TimescaleDB container.

### 2. Database layer — `src/synesis/storage/database.py`
- `store_raw_tweets(tweets: list[dict])` — bulk INSERT with ON CONFLICT DO NOTHING (dedup by unique constraint). Each dict has: `account_username`, `tweet_text`, `tweet_timestamp`, `tweet_url`
- `get_raw_tweets(since_hours: int = 24) -> list[dict]` — fetch recent tweets (for future use by intelligence pipeline)

### 3. Analyzer — `src/synesis/processing/twitter/analyzer.py`
- **Delete** `get_quote` tool
- **Delete** `get_options_snapshot` tool
- **Remove** `yfinance` from `TwitterAgentDeps` and the `YFinanceClient` TYPE_CHECKING import
- **Update** `SYSTEM_PROMPT` — remove references to getting quotes/options data. The LLM extracts from tweets + web research only.
- **Update** `analyze_tweets()`: Remove `yfinance` parameter

### 4. Job — `src/synesis/processing/twitter/job.py`
- **Add tweet storage step** before analysis:
  1. After fetching tweets from curated accounts (existing logic)
  2. Call `db.store_raw_tweets(tweets)` to persist
  3. Pass the same tweets to `analyzer.analyze_tweets()` as before
- Remove `yfinance: YFinanceClient | None = None` param from `twitter_agent_job()`
- Remove `YFinanceClient` TYPE_CHECKING import
- Remove `yfinance=yfinance` from `analyzer.analyze_tweets()` call

### 5. Agent entry point — `src/synesis/agent/__main__.py`
- Remove `YFinanceClient` import and `yf_client = YFinanceClient(redis=redis)` creation
- Update scheduler args: `[watchlist, yf_client, db]` → `[watchlist, db]`
- Update trigger function: `twitter_agent_job(watchlist, yf_client, db)` → `twitter_agent_job(watchlist, db)`

### 6. Tests — `tests/unit/test_twitter_intel.py` + `tests/integration/test_twitter_intel_e2e.py`
- Remove `test_job_passes_yfinance` test
- Remove `yfinance` param from all `analyze_tweets` calls and `twitter_agent_job` calls
- Add `raw_tweets` storage assertion (tweets saved before analysis)

## File Order (dependencies)
1. `database/init.sql` (schema)
2. `src/synesis/storage/database.py` (tweet CRUD)
3. `src/synesis/processing/twitter/analyzer.py` (remove yfinance tools)
4. `src/synesis/processing/twitter/job.py` (add tweet storage + remove yfinance param)
5. `src/synesis/agent/__main__.py` (entry point cleanup)
6. Tests

## Verification
1. `uv run ruff check --fix . && uv run ruff format .` — lint/format
2. `uv run mypy src/` — type check
3. `uv run pytest tests/unit/test_twitter_intel.py -v` — unit tests
4. `uv run pytest` — full test suite
5. Verify `raw_tweets` table populated after manual trigger
