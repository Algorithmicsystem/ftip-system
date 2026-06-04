CREATE TABLE IF NOT EXISTS cross_asset_snapshots (
    as_of_date                      DATE PRIMARY KEY,
    equity_regime                   TEXT,
    cross_asset_confirmation_score  NUMERIC(6, 2),
    regime_consistency              TEXT,
    fixed_income_signal             TEXT,
    currency_signal                 TEXT,
    commodity_signal                TEXT,
    volatility_signal               TEXT,
    equity_signal_amplifier         NUMERIC(6, 4),
    macro_narrative                 TEXT,
    meta                            JSONB NOT NULL DEFAULT '{}',
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
);
