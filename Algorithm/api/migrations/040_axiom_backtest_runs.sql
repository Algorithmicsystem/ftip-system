CREATE TABLE IF NOT EXISTS axiom_backtest_runs (
    run_id          TEXT        PRIMARY KEY,
    from_date       DATE        NOT NULL,
    to_date         DATE        NOT NULL,
    horizon_days    INT         NOT NULL,
    min_dau         NUMERIC     NOT NULL DEFAULT 0,
    signal_filter   TEXT[]      NOT NULL DEFAULT '{BUY,SELL}',
    long_only       BOOLEAN     NOT NULL DEFAULT TRUE,
    total_signals   INT,
    hit_rate        NUMERIC,
    avg_return_pct  NUMERIC,
    sharpe          NUMERIC,
    max_drawdown    NUMERIC,
    spearman_ic     NUMERIC,
    result          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_axiom_backtest_runs_created
    ON axiom_backtest_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_axiom_backtest_runs_dates
    ON axiom_backtest_runs (from_date DESC, to_date DESC);
