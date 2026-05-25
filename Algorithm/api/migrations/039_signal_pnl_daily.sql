CREATE TABLE IF NOT EXISTS signal_pnl_daily (
    symbol           TEXT        NOT NULL,
    signal_date      DATE        NOT NULL,
    horizon_days     INT         NOT NULL,
    lookback         INT         NOT NULL DEFAULT 252,
    signal_label     TEXT,
    signal_score     NUMERIC,
    dau              NUMERIC,
    regime_label     TEXT,
    price_at_signal  NUMERIC,
    horizon_date     DATE,
    price_at_horizon NUMERIC,
    return_pct       NUMERIC,
    hit              BOOLEAN,
    computed_at      DATE        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, signal_date, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_signal_pnl_signal_date
    ON signal_pnl_daily (signal_date DESC, symbol);

CREATE INDEX IF NOT EXISTS idx_signal_pnl_computed_at
    ON signal_pnl_daily (computed_at DESC);
