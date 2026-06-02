CREATE TABLE IF NOT EXISTS earnings_intelligence_daily (
    symbol                      TEXT        NOT NULL,
    as_of_date                  DATE        NOT NULL,
    pess                        NUMERIC,
    eis_trend_delta             NUMERIC,
    accruals_acceleration       NUMERIC,
    guidance_revision_velocity  NUMERIC,
    insider_sell_ratio          NUMERIC,
    days_to_earnings            INT,
    earnings_stress_flag        BOOLEAN     NOT NULL DEFAULT FALSE,
    meta                        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_earnings_intelligence_date ON earnings_intelligence_daily (as_of_date DESC, symbol);
