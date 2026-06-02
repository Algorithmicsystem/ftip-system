-- Migration 050: Company-Macro Sensitivity Table
-- Stores estimated sensitivity betas of company returns to macro factors.
-- Updated quarterly; feeds into state_pricing engine factor_compensation.

CREATE TABLE IF NOT EXISTS company_macro_sensitivity (
    symbol          TEXT        NOT NULL,
    macro_factor    TEXT        NOT NULL,   -- gdp_growth | inflation | credit_spread | vix | usd_index | oil_price | rates_10y
    sensitivity_beta NUMERIC    NOT NULL,   -- OLS beta of returns on factor
    r_squared       NUMERIC,               -- R² of regression
    lookback_days   INT         NOT NULL DEFAULT 252,
    estimated_at    DATE        NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, macro_factor, estimated_at)
);

CREATE INDEX IF NOT EXISTS idx_company_macro_sensitivity_symbol
    ON company_macro_sensitivity (symbol, estimated_at DESC);

CREATE INDEX IF NOT EXISTS idx_company_macro_sensitivity_factor
    ON company_macro_sensitivity (macro_factor, estimated_at DESC);

COMMENT ON TABLE company_macro_sensitivity IS
    'Quarterly OLS regression betas of equity returns on macro factors. '
    'Feeds state_pricing engine factor_compensation inputs.';
