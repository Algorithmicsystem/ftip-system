CREATE TABLE IF NOT EXISTS epa_violations (
    symbol               TEXT        NOT NULL,
    as_of_date           DATE        NOT NULL,
    violation_count_3yr  INTEGER     NOT NULL DEFAULT 0,
    total_penalties_usd  NUMERIC     NOT NULL DEFAULT 0,
    esg_risk_score       NUMERIC     NOT NULL DEFAULT 0,
    facilities_count     INTEGER     NOT NULL DEFAULT 0,
    source               TEXT        NOT NULL DEFAULT 'EPA_ECHO',
    raw                  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_epa_violations_symbol_date
    ON epa_violations(symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_epa_violations_esg_risk
    ON epa_violations(esg_risk_score DESC);
