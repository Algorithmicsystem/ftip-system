CREATE TABLE IF NOT EXISTS quality_daily (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    bars_ok BOOLEAN NOT NULL DEFAULT FALSE,
    fundamentals_ok BOOLEAN NOT NULL DEFAULT FALSE,
    sentiment_ok BOOLEAN NOT NULL DEFAULT FALSE,
    intraday_ok BOOLEAN NOT NULL DEFAULT FALSE,
    missingness NUMERIC NULL,
    anomaly_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_score INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_quality_daily_symbol_date ON quality_daily(symbol, as_of_date DESC);
