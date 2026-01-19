CREATE TABLE IF NOT EXISTS signals_intraday (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,
    action TEXT NOT NULL,
    score NUMERIC NOT NULL,
    confidence NUMERIC NOT NULL,
    reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    signal_version INT NOT NULL DEFAULT 1,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, ts, timeframe)
);

CREATE INDEX IF NOT EXISTS idx_signals_intraday_symbol_ts ON signals_intraday(symbol, ts DESC);
