-- Synesis Database Initialization
-- This script runs on first database creation

-- =============================================================================
-- SCHEMA SETUP
-- =============================================================================

-- Create dedicated schema for synesis tables
CREATE SCHEMA IF NOT EXISTS synesis;

-- Set search_path to synesis schema first, then public for extensions
SET search_path TO synesis, public;

-- =============================================================================
-- EXTENSIONS (installed in public schema)
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;  -- Time-series support

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Signals (TimescaleDB Hypertable)
-- Flow 1 (news) outputs with automatic time-based partitioning
-- flow_id: 'news'
-- signal_type: event type from extraction (e.g. 'earnings', 'macro', 'other')
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.signals (
    time TIMESTAMPTZ NOT NULL,
    flow_id TEXT NOT NULL,              -- 'news'
    payload JSONB NOT NULL,             -- Full signal data (NewsSignal JSON)
    markets JSONB,                      -- [{"market_id": "abc", "question": "...", ...}]
    tickers TEXT[],                     -- Matched tickers from Stage 1 ['NVDA', 'MRVL']
    entities TEXT[],                    -- All entities from Stage 2 LLM
    PRIMARY KEY (time, flow_id)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('synesis.signals', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for JSONB markets column
CREATE INDEX IF NOT EXISTS idx_signals_markets
    ON synesis.signals USING GIN (markets jsonb_path_ops);

-- Index for TEXT[] tickers column
CREATE INDEX IF NOT EXISTS idx_signals_tickers
    ON synesis.signals USING GIN (tickers);

-- Index for TEXT[] entities column
CREATE INDEX IF NOT EXISTS idx_signals_entities
    ON synesis.signals USING GIN (entities);

-- Enable compression after 7 days
ALTER TABLE synesis.signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'flow_id'
);
SELECT add_compression_policy('synesis.signals', INTERVAL '7 days', if_not_exists => TRUE);


-- -----------------------------------------------------------------------------
-- Predictions (TimescaleDB Hypertable)
-- Flow 1 Stage 2B prediction market evaluations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.predictions (
    time TIMESTAMPTZ NOT NULL,
    market_id TEXT NOT NULL,
    market_question TEXT NOT NULL,
    is_relevant BOOLEAN NOT NULL,
    verdict TEXT NOT NULL,           -- 'undervalued', 'overvalued', 'fair', 'skip'
    current_price DECIMAL(6,4),      -- Current YES price (0.0000 to 1.0000)
    estimated_fair_price DECIMAL(6,4), -- Estimated fair price
    edge DECIMAL(6,4),               -- fair - current
    confidence DECIMAL(5,4),         -- 0.0000 to 1.0000
    recommended_side TEXT,           -- 'yes', 'no', 'skip'
    reasoning TEXT,
    PRIMARY KEY (time, market_id)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('synesis.predictions', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for finding predictions with edge
CREATE INDEX IF NOT EXISTS idx_predictions_edge
    ON synesis.predictions (edge DESC NULLS LAST)
    WHERE is_relevant = TRUE AND edge IS NOT NULL;

-- Index for market-based queries
CREATE INDEX IF NOT EXISTS idx_predictions_market
    ON synesis.predictions (market_id, time DESC);

-- Enable compression after 7 days
ALTER TABLE synesis.predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'market_id, verdict'
);
SELECT add_compression_policy('synesis.predictions', INTERVAL '7 days', if_not_exists => TRUE);


-- -----------------------------------------------------------------------------
-- Raw Messages
-- Ingested messages from Telegram and Google News RSS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.raw_messages (
    id BIGSERIAL PRIMARY KEY,
    source_platform TEXT NOT NULL,      -- 'telegram', 'google_rss'
    source_account TEXT NOT NULL,       -- '@DeItaone', 'marketfeed'
    external_id TEXT NOT NULL,          -- Platform-specific ID
    raw_text TEXT NOT NULL,
    source_timestamp TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_platform, external_id)
);

-- Index for finding recent messages by source
CREATE INDEX IF NOT EXISTS idx_raw_messages_source_time
    ON synesis.raw_messages (source_platform, source_account, source_timestamp DESC);


-- -----------------------------------------------------------------------------
-- Watchlist
-- Tickers added by Flow 1 (news analysis) for tracking
-- added_by: source platform ('telegram', 'google_rss', 'manual')
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.watchlist (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    added_by TEXT NOT NULL,             -- 'telegram', 'google_rss', 'manual'
    added_reason TEXT,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,             -- TTL for auto-removal
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_watchlist_active
    ON synesis.watchlist (ticker) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_watchlist_expiry
    ON synesis.watchlist (expires_at) WHERE is_active = TRUE AND expires_at IS NOT NULL;


-- -----------------------------------------------------------------------------
-- Calendar Events
-- Event Radar: market-relevant events discovered from crawling + APIs
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.calendar_events (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    event_date DATE NOT NULL,
    event_end_date DATE,
    category TEXT NOT NULL,           -- earnings | economic_data | fed | 13f_filing |
                                      -- conference | release | regulatory | other
    sector TEXT,                       -- ai | energy | precious_metals | NULL
    region TEXT[] NOT NULL,            -- {'US','JP','SG','HK','global'}
    tickers TEXT[] DEFAULT '{}',
    source_urls TEXT[] NOT NULL DEFAULT '{}',
    time_label VARCHAR(10),
    discovered_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (title, event_date)
);

CREATE INDEX IF NOT EXISTS idx_cal_events_date
    ON synesis.calendar_events (event_date);
CREATE INDEX IF NOT EXISTS idx_cal_events_tickers
    ON synesis.calendar_events USING GIN (tickers);
CREATE INDEX IF NOT EXISTS idx_cal_events_region
    ON synesis.calendar_events USING GIN (region);
CREATE INDEX IF NOT EXISTS idx_cal_events_sector
    ON synesis.calendar_events (sector);
CREATE INDEX IF NOT EXISTS idx_cal_events_category
    ON synesis.calendar_events (category);


-- -----------------------------------------------------------------------------
-- Diary
-- Persisted pipeline outputs (twitter digest, event digest, market brief)
-- Keyed by (entry_date, source) so re-runs overwrite the same day's entry.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.diary (
    id SERIAL PRIMARY KEY,
    entry_date DATE NOT NULL,
    source TEXT NOT NULL,          -- 'twitter', 'events', 'market_brief'
    payload JSONB NOT NULL,        -- Full Pydantic model dump
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (entry_date, source)
);
CREATE INDEX IF NOT EXISTS idx_diary_source ON synesis.diary (source, entry_date DESC);


-- =============================================================================
-- GRANTS (for application user)
-- =============================================================================

-- These would be run separately with actual user names
-- GRANT USAGE ON SCHEMA synesis TO synesis_app;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA synesis TO synesis_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA synesis TO synesis_app;
