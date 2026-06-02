CREATE TABLE IF NOT EXISTS institutional_flow_daily (
    symbol                  TEXT        NOT NULL,
    as_of_date              DATE        NOT NULL,
    ias                     NUMERIC,
    dark_pool_buy_ratio     NUMERIC,
    block_direction         NUMERIC,
    short_interest_change   NUMERIC,
    meta                    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_institutional_flow_date ON institutional_flow_daily (as_of_date DESC, symbol);
