-- Migration 053: Private Entity Financials
-- Periodic financial snapshots for PE portfolio companies.

CREATE TABLE IF NOT EXISTS private_entity_financials (
    entity_id       TEXT        NOT NULL,
    period_end      DATE        NOT NULL,
    period_type     TEXT        NOT NULL DEFAULT 'quarterly',  -- quarterly | annual | ttm
    revenue         NUMERIC,                   -- total revenue ($M)
    ebitda          NUMERIC,                   -- EBITDA ($M)
    net_income      NUMERIC,
    total_debt      NUMERIC,
    cash            NUMERIC,
    capex           NUMERIC,
    free_cash_flow  NUMERIC,                   -- operating CF - capex
    headcount       INT,
    arr             NUMERIC,                   -- annual recurring revenue (SaaS)
    reported_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, period_end, period_type)
);

CREATE INDEX IF NOT EXISTS idx_pef_entity_period
    ON private_entity_financials (entity_id, period_end DESC);

COMMENT ON TABLE private_entity_financials IS
    'Periodic financial snapshots for PE portfolio companies. '
    'Supports quarterly, annual, and TTM period_type values. '
    'arr column captures SaaS / subscription recurring revenue.';
