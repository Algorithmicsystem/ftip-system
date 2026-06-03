CREATE TABLE IF NOT EXISTS family_office_portfolios (
    portfolio_id            TEXT PRIMARY KEY,
    family_office_name      TEXT NOT NULL,
    as_of_date              DATE NOT NULL,
    total_value_usd         NUMERIC(20, 4) NOT NULL,
    positions               JSONB NOT NULL DEFAULT '[]',
    asset_class_allocation  JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fop_family_date
    ON family_office_portfolios (family_office_name, as_of_date DESC);
