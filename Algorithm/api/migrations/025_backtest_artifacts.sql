CREATE TABLE IF NOT EXISTS backtest_artifacts (
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_backtest_artifacts_run ON backtest_artifacts(run_id);

