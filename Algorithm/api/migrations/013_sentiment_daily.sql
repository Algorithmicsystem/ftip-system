CREATE TABLE IF NOT EXISTS sentiment_daily (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    headline_count INT NOT NULL DEFAULT 0,
    sentiment_mean NUMERIC NULL,
    sentiment_pos INT NOT NULL DEFAULT 0,
    sentiment_neg INT NOT NULL DEFAULT 0,
    sentiment_neu INT NOT NULL DEFAULT 0,
    sentiment_score NUMERIC NULL,
    source TEXT NOT NULL DEFAULT 'llm_v1',
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_daily_symbol_date ON sentiment_daily(symbol, as_of_date DESC);
