CREATE TABLE IF NOT EXISTS narrative_intelligence_daily (
    symbol                TEXT        NOT NULL,
    as_of_date            DATE        NOT NULL,
    nss                   NUMERIC,
    nms                   NUMERIC,
    inflection_flag       BOOLEAN     NOT NULL DEFAULT FALSE,
    inflection_direction  TEXT,
    dominant_narrative    TEXT,
    article_count         INT,
    meta                  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_narrative_intelligence_date ON narrative_intelligence_daily (as_of_date DESC, symbol);
