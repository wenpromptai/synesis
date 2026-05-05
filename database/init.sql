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
-- Watchlist
-- Tickers tracked for analysis purposes
-- added_by: source ('manual', 'intelligence', 'twitter')
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.watchlist (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    added_by TEXT NOT NULL,
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
-- Raw Tweets
-- Twitter/X tweets fetched daily from curated accounts, persisted before LLM analysis.
-- Used by the Twitter agent job and future intelligence pipeline.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synesis.raw_tweets (
    tweet_id TEXT NOT NULL,
    account_username TEXT NOT NULL,
    tweet_text TEXT NOT NULL,
    tweet_timestamp TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tweet_url TEXT,
    PRIMARY KEY (account_username, tweet_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_tweets_fetched
    ON synesis.raw_tweets (fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_tweets_account
    ON synesis.raw_tweets (account_username, tweet_timestamp DESC);


-- =============================================================================
-- GRANTS (for application user)
-- =============================================================================

-- These would be run separately with actual user names
-- GRANT USAGE ON SCHEMA synesis TO synesis_app;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA synesis TO synesis_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA synesis TO synesis_app;
