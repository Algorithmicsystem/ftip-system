-- Migration 110: Sector briefings table for tiered morning briefing
CREATE TABLE IF NOT EXISTS axiom_sector_briefings (
    id             SERIAL PRIMARY KEY,
    briefing_date  DATE        NOT NULL,
    sector         TEXT        NOT NULL,
    brief_text     TEXT        NOT NULL,
    signal_summary JSONB,
    generated_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (briefing_date, sector)
);

CREATE INDEX IF NOT EXISTS idx_sector_briefings_date
    ON axiom_sector_briefings(briefing_date DESC);
