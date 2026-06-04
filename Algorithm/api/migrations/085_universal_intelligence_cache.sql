CREATE TABLE IF NOT EXISTS universal_intelligence_cache (
    symbol          TEXT NOT NULL,
    as_of_date      DATE NOT NULL,
    response        JSONB NOT NULL DEFAULT '{}',
    assembled_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_uic_symbol_date
    ON universal_intelligence_cache (symbol, as_of_date DESC);
