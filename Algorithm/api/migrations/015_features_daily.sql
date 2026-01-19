CREATE TABLE IF NOT EXISTS features_daily (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    ret_1d NUMERIC NULL,
    ret_5d NUMERIC NULL,
    ret_21d NUMERIC NULL,
    vol_21d NUMERIC NULL,
    vol_63d NUMERIC NULL,
    atr_14 NUMERIC NULL,
    atr_pct NUMERIC NULL,
    trend_slope_21d NUMERIC NULL,
    trend_r2_21d NUMERIC NULL,
    trend_slope_63d NUMERIC NULL,
    trend_r2_63d NUMERIC NULL,
    mom_vol_adj_21d NUMERIC NULL,
    maxdd_63d NUMERIC NULL,
    dollar_vol_21d NUMERIC NULL,
    sentiment_score NUMERIC NULL,
    sentiment_surprise NUMERIC NULL,
    regime_label TEXT NULL,
    regime_strength NUMERIC NULL,
    feature_version INT NOT NULL DEFAULT 1,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_features_daily_symbol_date ON features_daily(symbol, as_of_date DESC);
