-- Synesis Database Initialization
-- This script runs on first database creation

-- Enable required extensions for vector search and text matching
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- TimescaleDB extension is pre-installed in the timescaledb image
-- but we ensure it's enabled for our database
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
