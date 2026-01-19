CREATE TABLE IF NOT EXISTS market_bars_intraday (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,
    open NUMERIC NULL,
    high NUMERIC NULL,
    low NUMERIC NULL,
    close NUMERIC NULL,
    volume BIGINT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, ts, timeframe)
);

CREATE INDEX IF NOT EXISTS idx_market_bars_intraday_symbol_ts ON market_bars_intraday(symbol, ts DESC);
