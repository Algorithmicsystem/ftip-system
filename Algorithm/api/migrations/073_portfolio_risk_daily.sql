CREATE TABLE IF NOT EXISTS portfolio_risk_daily (
    portfolio_id          TEXT        NOT NULL,
    as_of_date            DATE        NOT NULL,
    var_99_1d             NUMERIC,
    cvar_99_1d            NUMERIC,
    var_99_horizon        NUMERIC,
    horizon_days          INT         NOT NULL DEFAULT 5,
    marginal_var          JSONB,
    diversification_benefit NUMERIC,
    concentration_risk    BOOLEAN,
    methodology           TEXT,
    meta                  JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (portfolio_id, as_of_date)
);
