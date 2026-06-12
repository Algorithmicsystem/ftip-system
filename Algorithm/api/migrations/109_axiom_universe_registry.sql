-- Migration 109: AXIOM Universe Registry
-- Replaces the hardcoded AXIOM_UNIVERSE list with a DB-driven
-- symbol registry supporting 10,000 symbols with tier classification.

CREATE TABLE IF NOT EXISTS axiom_universe_registry (
    symbol           TEXT        PRIMARY KEY,
    company_name     TEXT,
    sector           TEXT,
    industry         TEXT,
    country          TEXT        DEFAULT 'US',
    exchange         TEXT,
    market_cap_usd   BIGINT,
    tier             SMALLINT    NOT NULL DEFAULT 3,
    active           BOOLEAN     NOT NULL DEFAULT TRUE,
    last_validated   DATE,
    avg_daily_volume BIGINT,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_universe_tier
    ON axiom_universe_registry(tier) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_universe_sector
    ON axiom_universe_registry(sector) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_universe_market_cap
    ON axiom_universe_registry(market_cap_usd DESC NULLS LAST) WHERE active = TRUE;
