CREATE TABLE IF NOT EXISTS backtest_runs (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    universe TEXT NOT NULL,
    symbol TEXT NULL,
    date_start DATE NOT NULL,
    date_end DATE NOT NULL,
    horizon TEXT NOT NULL,
    risk_mode TEXT NOT NULL,
    signal_version_hash TEXT NOT NULL,
    cost_model JSONB NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    entry_dt DATE NOT NULL,
    exit_dt DATE NOT NULL,
    side TEXT NOT NULL,
    entry_px DOUBLE PRECISION NOT NULL,
    exit_px DOUBLE PRECISION NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    pnl DOUBLE PRECISION NOT NULL,
    pnl_pct DOUBLE PRECISION NOT NULL,
    holding_days INT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_equity_curve (
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    dt DATE NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    drawdown DOUBLE PRECISION NOT NULL,
    benchmark_equity DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    run_id UUID PRIMARY KEY REFERENCES backtest_runs(id) ON DELETE CASCADE,
    cagr DOUBLE PRECISION NOT NULL,
    sharpe DOUBLE PRECISION NOT NULL,
    sortino DOUBLE PRECISION NOT NULL,
    maxdd DOUBLE PRECISION NOT NULL,
    volatility DOUBLE PRECISION NOT NULL,
    winrate DOUBLE PRECISION NOT NULL,
    avgtrade DOUBLE PRECISION NOT NULL,
    tradesperyear DOUBLE PRECISION NOT NULL,
    turnover DOUBLE PRECISION NOT NULL,
    alpha_vs_spy DOUBLE PRECISION NULL,
    beta DOUBLE PRECISION NULL
);

CREATE TABLE IF NOT EXISTS backtest_regime_metrics (
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    regime_name TEXT NOT NULL,
    cagr DOUBLE PRECISION NOT NULL,
    sharpe DOUBLE PRECISION NOT NULL,
    maxdd DOUBLE PRECISION NOT NULL,
    winrate DOUBLE PRECISION NOT NULL,
    trades INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_created ON backtest_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_equity_curve_run_dt ON backtest_equity_curve(run_id, dt);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_regime_metrics_run ON backtest_regime_metrics(run_id);
