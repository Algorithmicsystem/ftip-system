CREATE TABLE IF NOT EXISTS macro_intelligence_snapshots (
    as_of_date              DATE PRIMARY KEY,
    gdp_regime              TEXT,
    inflation_regime        TEXT,
    monetary_regime         TEXT,
    credit_regime           TEXT,
    equity_macro_score      NUMERIC(6, 2),
    macro_environment_score NUMERIC(6, 2),
    favored_axiom_factors   JSONB NOT NULL DEFAULT '[]',
    unfavored_axiom_factors JSONB NOT NULL DEFAULT '[]',
    macro_regime_label      TEXT,
    meta                    JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
