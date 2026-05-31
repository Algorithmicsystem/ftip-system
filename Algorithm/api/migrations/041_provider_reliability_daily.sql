CREATE TABLE IF NOT EXISTS provider_reliability_daily (
    as_of_date      DATE        NOT NULL,
    provider        TEXT        NOT NULL,   -- finnhub, fred, sec_edgar, etc.
    is_enabled      BOOLEAN     NOT NULL DEFAULT TRUE,
    status          TEXT        NOT NULL,   -- ok | degraded | down
    message         TEXT        NOT NULL DEFAULT '',
    meta            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (as_of_date, provider)
);

CREATE INDEX IF NOT EXISTS idx_provider_reliability_date
    ON provider_reliability_daily (as_of_date DESC, provider);

CREATE INDEX IF NOT EXISTS idx_provider_reliability_provider
    ON provider_reliability_daily (provider, as_of_date DESC);
