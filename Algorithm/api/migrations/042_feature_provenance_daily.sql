CREATE TABLE IF NOT EXISTS feature_provenance_daily (
    symbol              TEXT        NOT NULL,
    as_of               DATE        NOT NULL,
    feature_version     TEXT        NOT NULL,
    lookback            INT         NOT NULL DEFAULT 63,
    coverage_status     TEXT        NOT NULL DEFAULT 'unknown',
    price_source        TEXT,
    event_source        TEXT,
    breadth_source      TEXT,
    sentiment_source    TEXT,
    null_feature_count  INT         NOT NULL DEFAULT 0,
    total_feature_count INT         NOT NULL DEFAULT 0,
    missing_features    TEXT[]      NOT NULL DEFAULT '{}',
    data_warnings       TEXT[]      NOT NULL DEFAULT '{}',
    feature_hash        TEXT,
    snapshot_id         TEXT,
    meta                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of, feature_version)
);

CREATE INDEX IF NOT EXISTS idx_feature_provenance_symbol_asof
    ON feature_provenance_daily (symbol, as_of DESC);

CREATE INDEX IF NOT EXISTS idx_feature_provenance_coverage
    ON feature_provenance_daily (as_of DESC, coverage_status);
