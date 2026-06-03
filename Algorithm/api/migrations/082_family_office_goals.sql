CREATE TABLE IF NOT EXISTS family_office_goals (
    goal_id                 TEXT PRIMARY KEY,
    portfolio_id            TEXT NOT NULL,
    goal_type               TEXT NOT NULL,
    label                   TEXT NOT NULL,
    target_amount_usd       NUMERIC(20, 4) NOT NULL,
    target_date_years       NUMERIC(6, 2) NOT NULL,
    required_return_annual  NUMERIC(8, 6) NOT NULL,
    risk_budget             NUMERIC(8, 6) NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fog_portfolio
    ON family_office_goals (portfolio_id);
