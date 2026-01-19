CREATE TABLE IF NOT EXISTS features_intraday (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL,
    ret_1bar NUMERIC NULL,
    vol_n NUMERIC NULL,
    trend_slope_n NUMERIC NULL,
    feature_version INT NOT NULL DEFAULT 1,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, ts, timeframe)
);

CREATE INDEX IF NOT EXISTS idx_features_intraday_symbol_ts ON features_intraday(symbol, ts DESC);
