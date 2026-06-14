CREATE TABLE IF NOT EXISTS litigation_risk (
    symbol                  TEXT        NOT NULL,
    as_of_date              DATE        NOT NULL,
    active_cases_1yr        INTEGER     NOT NULL DEFAULT 0,
    securities_fraud_cases  INTEGER     NOT NULL DEFAULT 0,
    employment_cases        INTEGER     NOT NULL DEFAULT 0,
    antitrust_cases         INTEGER     NOT NULL DEFAULT 0,
    ip_cases                INTEGER     NOT NULL DEFAULT 0,
    other_cases             INTEGER     NOT NULL DEFAULT 0,
    total_litigation_score  NUMERIC     NOT NULL DEFAULT 0,
    source                  TEXT        NOT NULL DEFAULT 'CourtListener',
    raw                     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_litigation_risk_symbol_date
    ON litigation_risk(symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_litigation_risk_score
    ON litigation_risk(total_litigation_score DESC);
