CREATE TABLE IF NOT EXISTS axiom_intraday_updates (
    symbol                    TEXT NOT NULL,
    update_time               TIMESTAMPTZ NOT NULL,
    session_date              DATE NOT NULL,
    intraday_flow_score       NUMERIC,
    intraday_behavioral_score NUMERIC,
    intraday_composite        NUMERIC,
    vwap_deviation            NUMERIC,
    intraday_momentum_score   NUMERIC,
    volume_surge_score        NUMERIC,
    session_return_pct        NUMERIC,
    alert_eligible            BOOLEAN NOT NULL DEFAULT FALSE,
    meta                      JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (symbol, update_time)
);

CREATE INDEX IF NOT EXISTS idx_axiom_intraday_updates_session
    ON axiom_intraday_updates (session_date, symbol, update_time DESC);
