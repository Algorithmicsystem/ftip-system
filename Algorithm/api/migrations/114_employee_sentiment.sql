CREATE TABLE IF NOT EXISTS employee_sentiment (
    symbol         TEXT        NOT NULL,
    as_of_date     DATE        NOT NULL,
    overall_rating NUMERIC,
    ceo_approval   NUMERIC,
    culture_score  NUMERIC,
    source         TEXT        NOT NULL DEFAULT 'neutral_fallback',
    raw            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_employee_sentiment_symbol_date
    ON employee_sentiment(symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_employee_sentiment_date
    ON employee_sentiment(as_of_date DESC);
