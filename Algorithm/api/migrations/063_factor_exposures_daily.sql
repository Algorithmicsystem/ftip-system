CREATE TABLE IF NOT EXISTS factor_exposures_daily (
    symbol             TEXT        NOT NULL,
    as_of_date         DATE        NOT NULL,
    regime_label       TEXT,
    factor_name        TEXT        NOT NULL,
    loading            NUMERIC,
    t_stat             NUMERIC,
    theoretical_source TEXT,
    meta               JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_factor_exposures_screening
    ON factor_exposures_daily (as_of_date DESC, factor_name, loading DESC);
