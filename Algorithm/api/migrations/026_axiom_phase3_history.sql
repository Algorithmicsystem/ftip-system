CREATE TABLE IF NOT EXISTS axiom_scores_daily (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    framework_version TEXT NOT NULL,
    snapshot_id TEXT,
    snapshot_version TEXT,
    feature_version TEXT,
    signal_version TEXT,
    regime_label TEXT,
    trade_family TEXT,
    deployability_tier TEXT,
    evidence_backed_deployability_tier TEXT,
    size_band TEXT,
    gross_opportunity DOUBLE PRECISION,
    friction_burden DOUBLE PRECISION,
    validated_edge DOUBLE PRECISION,
    deployable_alpha_utility DOUBLE PRECISION,
    overall_coverage DOUBLE PRECISION,
    overall_confidence DOUBLE PRECISION,
    payload JSONB NOT NULL,
    outcome_payload JSONB,
    build_meta JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date, framework_version)
);

CREATE INDEX IF NOT EXISTS idx_axiom_scores_daily_as_of
    ON axiom_scores_daily(as_of_date DESC, symbol);

CREATE INDEX IF NOT EXISTS idx_axiom_scores_daily_regime
    ON axiom_scores_daily(regime_label, trade_family, deployability_tier);

CREATE TABLE IF NOT EXISTS axiom_replay_runs (
    run_id TEXT PRIMARY KEY,
    symbols JSONB NOT NULL,
    date_start DATE NOT NULL,
    date_end DATE NOT NULL,
    lookback INT NOT NULL,
    framework_version TEXT NOT NULL,
    record_count INT NOT NULL DEFAULT 0,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_axiom_replay_runs_dates
    ON axiom_replay_runs(date_start, date_end, created_at DESC);

CREATE TABLE IF NOT EXISTS axiom_calibration_snapshots (
    snapshot_key TEXT PRIMARY KEY,
    as_of_date DATE,
    horizon_label TEXT NOT NULL,
    framework_version TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_axiom_calibration_snapshots_as_of
    ON axiom_calibration_snapshots(as_of_date DESC, horizon_label);
