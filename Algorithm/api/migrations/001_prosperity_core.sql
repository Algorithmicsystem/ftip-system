CREATE TABLE IF NOT EXISTS prosperity_universe (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    exchange TEXT,
    asset_type TEXT,
    active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prosperity_daily_bars (
    symbol TEXT REFERENCES prosperity_universe(symbol),
    date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION NOT NULL,
    adj_close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    source TEXT NOT NULL,
    raw JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS prosperity_features_daily (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    lookback INT NOT NULL,
    mom_5 DOUBLE PRECISION,
    mom_21 DOUBLE PRECISION,
    mom_63 DOUBLE PRECISION,
    trend_sma20_50 DOUBLE PRECISION,
    volatility_ann DOUBLE PRECISION,
    rsi14 DOUBLE PRECISION,
    volume_z20 DOUBLE PRECISION,
    last_close DOUBLE PRECISION,
    regime TEXT,
    features_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date, lookback)
);

CREATE TABLE IF NOT EXISTS prosperity_signals_daily (
    symbol TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    lookback INT NOT NULL,
    score_mode TEXT NOT NULL,
    score DOUBLE PRECISION,
    base_score DOUBLE PRECISION,
    stacked_score DOUBLE PRECISION,
    thresholds JSONB NOT NULL,
    signal TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    regime TEXT,
    calibration_meta JSONB,
    notes JSONB,
    signal_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date, lookback, score_mode)
);

CREATE TABLE IF NOT EXISTS prosperity_backtest_runs (
    run_id UUID PRIMARY KEY,
    kind TEXT NOT NULL,
    request JSONB NOT NULL,
    response JSONB NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prosperity_calibrations (
    symbol TEXT NOT NULL,
    created_at_utc TEXT,
    payload JSONB NOT NULL,
    optimize_horizon INT,
    train_range JSONB,
    calibration_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, calibration_hash)
);

CREATE TABLE IF NOT EXISTS prosperity_events (
    id UUID PRIMARY KEY,
    event_type TEXT,
    symbol TEXT,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
