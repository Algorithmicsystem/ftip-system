CREATE TABLE IF NOT EXISTS market_bars_daily (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    open NUMERIC NULL,
    high NUMERIC NULL,
    low NUMERIC NULL,
    close NUMERIC NULL,
    volume BIGINT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_market_bars_daily_symbol_date ON market_bars_daily(symbol, as_of_date DESC);
