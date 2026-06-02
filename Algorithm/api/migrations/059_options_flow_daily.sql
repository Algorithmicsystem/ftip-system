CREATE TABLE IF NOT EXISTS options_flow_daily (
    symbol              TEXT        NOT NULL,
    as_of_date          DATE        NOT NULL,
    osms                NUMERIC,
    iv_skew             NUMERIC,
    pcr                 NUMERIC,
    unusual_volume_flag BOOLEAN     NOT NULL DEFAULT FALSE,
    call_volume         NUMERIC,
    put_volume          NUMERIC,
    avg_iv_atm          NUMERIC,
    large_block_pct     NUMERIC,
    meta                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_options_flow_daily_date ON options_flow_daily (as_of_date DESC, symbol);
