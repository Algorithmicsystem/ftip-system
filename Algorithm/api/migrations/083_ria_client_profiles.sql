CREATE TABLE IF NOT EXISTS ria_client_profiles (
    client_id               TEXT PRIMARY KEY,
    client_name             TEXT NOT NULL,
    portfolio_value_usd     NUMERIC(20, 4) NOT NULL,
    risk_tolerance          TEXT NOT NULL,
    time_horizon_years      NUMERIC(6, 2) NOT NULL,
    income_need_annual      NUMERIC(20, 4) NOT NULL DEFAULT 0,
    tax_bracket             NUMERIC(6, 4) NOT NULL,
    esg_preference          BOOLEAN NOT NULL DEFAULT FALSE,
    axiom_score             NUMERIC(6, 2),
    last_review_date        DATE,
    sri_score               NUMERIC(6, 2),
    metadata                JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rcp_review_date
    ON ria_client_profiles (last_review_date DESC NULLS FIRST);
