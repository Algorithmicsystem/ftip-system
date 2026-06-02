-- Migration 055: SMB Monthly Financials
-- Monthly operating snapshots for cash flow forecasting and supplier risk.

CREATE TABLE IF NOT EXISTS smb_monthly_financials (
    entity_id           TEXT        NOT NULL,
    month_end           DATE        NOT NULL,   -- last day of the month
    revenue             NUMERIC,                -- ($K)
    cogs                NUMERIC,                -- cost of goods sold ($K)
    operating_expenses  NUMERIC,                -- SG&A + other opex ($K)
    net_income          NUMERIC,
    cash_balance        NUMERIC,                -- end-of-month cash ($K)
    accounts_receivable NUMERIC,
    accounts_payable    NUMERIC,
    inventory           NUMERIC,
    payroll             NUMERIC,
    top_supplier_concentration NUMERIC,         -- % of COGS from single largest supplier (0–1)
    reported_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, month_end)
);

CREATE INDEX IF NOT EXISTS idx_smb_mf_entity_month
    ON smb_monthly_financials (entity_id, month_end DESC);

COMMENT ON TABLE smb_monthly_financials IS
    'Monthly financial snapshots for SMB entities. '
    'Feeds cash flow forecasting (12-month horizon) and supplier concentration risk.';
