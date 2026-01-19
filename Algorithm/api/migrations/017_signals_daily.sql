CREATE TABLE IF NOT EXISTS signals_daily (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    action TEXT NOT NULL CHECK(action in ('BUY','SELL','HOLD')),
    score NUMERIC NOT NULL,
    confidence NUMERIC NOT NULL,
    entry_low NUMERIC NULL,
    entry_high NUMERIC NULL,
    stop_loss NUMERIC NULL,
    take_profit_1 NUMERIC NULL,
    take_profit_2 NUMERIC NULL,
    horizon_days INT NOT NULL DEFAULT 21,
    reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    reason_details JSONB NOT NULL DEFAULT '{}'::jsonb,
    signal_version INT NOT NULL DEFAULT 1,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_signals_daily_symbol_date ON signals_daily(symbol, as_of_date DESC);
