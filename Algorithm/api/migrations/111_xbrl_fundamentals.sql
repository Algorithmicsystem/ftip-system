-- Migration 111: XBRL fundamentals cache table
CREATE TABLE IF NOT EXISTS xbrl_fundamentals (
    symbol          TEXT NOT NULL,
    cik             TEXT NOT NULL,
    fiscal_quarter  TEXT NOT NULL,
    metrics         JSONB NOT NULL DEFAULT '{}'::jsonb,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, fiscal_quarter)
);

CREATE INDEX IF NOT EXISTS idx_xbrl_fundamentals_symbol
    ON xbrl_fundamentals(symbol, fiscal_quarter DESC);

CREATE INDEX IF NOT EXISTS idx_xbrl_fundamentals_fetched
    ON xbrl_fundamentals(fetched_at DESC);
