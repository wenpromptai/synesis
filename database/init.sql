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
CREATE EXTENSION IF NOT EXISTS vector;           -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- Trigram for text search
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;  -- Time-series support

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Signals (TimescaleDB Hypertable)
-- Stores all flow outputs with automatic time-based partitioning
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.signals (
    time TIMESTAMPTZ NOT NULL,
    flow_id TEXT NOT NULL,              -- 'news', 'sentiment', 'market_intel'
    signal_type TEXT NOT NULL,          -- 'breaking_news', 'sentiment', 'market_intel'
    payload JSONB NOT NULL,             -- Full signal data
    markets JSONB,                      -- [{"market_id": "abc", "link": "https://..."}]
    tickers TEXT[],                     -- ['AAPL', 'TSLA']
    entities TEXT[],                    -- All entities mentioned (people, companies, institutions)
    PRIMARY KEY (time, flow_id)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable('synesis.signals', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- GIN Indexes for efficient JSONB and array queries
-- -----------------------------------------------------------------------------

-- Index for JSONB markets column
-- Uses jsonb_path_ops: smaller index, supports @> containment queries
CREATE INDEX IF NOT EXISTS idx_signals_markets
    ON synesis.signals USING GIN (markets jsonb_path_ops);

-- Index for TEXT[] tickers column
-- Supports: @> (contains), && (overlap), <@ (contained by)
CREATE INDEX IF NOT EXISTS idx_signals_tickers
    ON synesis.signals USING GIN (tickers);

-- Index for TEXT[] entities column
-- Supports: @> (contains), && (overlap), <@ (contained by)
CREATE INDEX IF NOT EXISTS idx_signals_entities
    ON synesis.signals USING GIN (entities);

-- Index for querying evaluations in payload (Flow 1 odds evaluations)
-- Supports queries like: payload->'evaluations' @> '[{"verdict": "undervalued"}]'
CREATE INDEX IF NOT EXISTS idx_signals_evaluations
    ON synesis.signals USING GIN ((payload->'evaluations') jsonb_path_ops);


-- Enable compression after 7 days
ALTER TABLE synesis.signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'flow_id, signal_type'
);
SELECT add_compression_policy('synesis.signals', INTERVAL '7 days', if_not_exists => TRUE);


-- -----------------------------------------------------------------------------
-- Predictions (TimescaleDB Hypertable)
-- Stores Stage 2B prediction market evaluations
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
-- Ingested messages from Twitter/Telegram with embeddings for dedup
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.raw_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_platform TEXT NOT NULL,      -- 'twitter', 'telegram'
    source_account TEXT NOT NULL,       -- '@DeItaone', 'marketfeed'
    source_type TEXT NOT NULL,          -- 'news', 'analysis'
    external_id TEXT NOT NULL,          -- Platform-specific ID
    raw_text TEXT NOT NULL,
    source_timestamp TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_platform, external_id)
);

-- Index for finding recent messages by source
CREATE INDEX IF NOT EXISTS idx_raw_messages_source_time
    ON synesis.raw_messages (source_platform, source_account, source_timestamp DESC);

-- =============================================================================
-- PREDICTION MARKETS (Flow 3)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Wallets
-- Tracked wallets on prediction market platforms
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.wallets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,             -- 'polymarket', 'kalshi'
    address TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_watched BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (platform, address)
);

CREATE INDEX IF NOT EXISTS idx_wallets_watched
    ON synesis.wallets (platform) WHERE is_watched = TRUE;

-- -----------------------------------------------------------------------------
-- Wallet Metrics
-- Aggregated performance metrics per wallet
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.wallet_metrics (
    wallet_id UUID PRIMARY KEY REFERENCES synesis.wallets(id) ON DELETE CASCADE,
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),              -- 0.0000 to 1.0000
    total_pnl DECIMAL(20,6) DEFAULT 0,
    -- Computed insider score
    insider_score DECIMAL(5,4),         -- 0.0 to 1.0
    -- Metadata
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for finding profitable wallets
CREATE INDEX IF NOT EXISTS idx_wallet_metrics_profitable
    ON synesis.wallet_metrics (win_rate DESC, insider_score DESC)
    WHERE total_trades >= 10;

-- =============================================================================
-- FLOW 3: MARKET INTELLIGENCE - SNAPSHOTS
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Market Snapshots (TimescaleDB Hypertable)
-- Volume/price history for analysis
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.market_snapshots (
    time TIMESTAMPTZ NOT NULL,
    platform TEXT NOT NULL,
    market_external_id TEXT NOT NULL,
    category TEXT,                      -- Market category
    yes_price DECIMAL(6,4),
    no_price DECIMAL(6,4),
    volume_1h DECIMAL(20,6),            -- Real WS-accumulated hourly volume (NULL if no WS)
    volume_24h DECIMAL(20,6),           -- 24h volume from REST API
    volume_total DECIMAL(20,6),         -- All-time total volume
    trade_count_1h INTEGER,             -- Trade count in last hour
    open_interest DECIMAL(20,6),
    PRIMARY KEY (time, platform, market_external_id)
);

SELECT create_hypertable('synesis.market_snapshots', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Compression after 7 days
ALTER TABLE synesis.market_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'platform, market_external_id'
);
SELECT add_compression_policy('synesis.market_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention: 90 days
SELECT add_retention_policy('synesis.market_snapshots', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================================
-- FLOW 2: SENTIMENT
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Watchlist
-- Tickers being monitored for sentiment
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.watchlist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker TEXT NOT NULL UNIQUE,
    company_name TEXT,
    added_by TEXT NOT NULL,             -- 'news', 'reddit', 'manual'
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
-- Sentiment Snapshots
-- Per-ticker sentiment at each analysis interval
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.sentiment_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker TEXT NOT NULL,
    snapshot_time TIMESTAMPTZ NOT NULL,
    -- Sentiment ratios
    bullish_ratio DECIMAL(5,4) NOT NULL,
    bearish_ratio DECIMAL(5,4) NOT NULL,
    -- Volume metrics
    mention_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time
    ON synesis.sentiment_snapshots (ticker, snapshot_time DESC);

-- =============================================================================
-- GRANTS (for application user)
-- =============================================================================

-- These would be run separately with actual user names
-- GRANT USAGE ON SCHEMA synesis TO synesis_app;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA synesis TO synesis_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA synesis TO synesis_app;
