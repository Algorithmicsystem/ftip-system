-- Alert rules: what conditions to watch per symbol
CREATE TABLE IF NOT EXISTS signal_alert_rules (
    rule_id                     TEXT        PRIMARY KEY,
    symbol                      TEXT        NOT NULL,
    min_dau                     NUMERIC     NOT NULL DEFAULT 65,
    signal_filter               TEXT[],                        -- NULL = any; e.g. '{BUY}' or '{BUY,SELL}'
    favorable_regimes           TEXT[],                        -- NULL = system defaults
    require_breadth_alignment   BOOLEAN     NOT NULL DEFAULT true,
    min_conviction_score        NUMERIC     NOT NULL DEFAULT 35,
    webhook_url                 TEXT,
    is_active                   BOOLEAN     NOT NULL DEFAULT true,
    meta                        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_signal_alert_rules_symbol
    ON signal_alert_rules (symbol, is_active);

-- Fired alert events (one per rule per day)
CREATE TABLE IF NOT EXISTS signal_alert_events (
    event_id            TEXT        PRIMARY KEY,
    rule_id             TEXT        NOT NULL REFERENCES signal_alert_rules(rule_id) ON DELETE CASCADE,
    symbol              TEXT        NOT NULL,
    as_of_date          DATE        NOT NULL,
    signal_label        TEXT        NOT NULL,
    dau                 NUMERIC,
    regime_label        TEXT,
    breadth_state       TEXT,
    ic_state            TEXT,
    conviction_score    NUMERIC,
    webhook_delivered   BOOLEAN     NOT NULL DEFAULT false,
    webhook_status_code INT,
    payload             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (rule_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_signal_alert_events_date
    ON signal_alert_events (as_of_date DESC, symbol);

CREATE INDEX IF NOT EXISTS idx_signal_alert_events_symbol
    ON signal_alert_events (symbol, as_of_date DESC);
