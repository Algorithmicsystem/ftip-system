CREATE TABLE IF NOT EXISTS factor_returns_daily (
    as_of_date      DATE        NOT NULL,
    factor_name     TEXT        NOT NULL,
    long_count      INT,
    short_count     INT,
    factor_return_pct NUMERIC,
    factor_ir       NUMERIC,
    meta            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (as_of_date, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_factor_returns_daily_date
    ON factor_returns_daily (as_of_date DESC, factor_name);
