-- Phase 28: regime transition event log
-- Stores day-over-day dominant regime shifts detected from axiom_scores_daily.

CREATE TABLE IF NOT EXISTS regime_transitions (
    transition_id   TEXT        NOT NULL PRIMARY KEY,
    as_of_date      DATE        NOT NULL,
    from_regime     TEXT        NOT NULL,
    to_regime       TEXT        NOT NULL,
    symbol_count    INT         NOT NULL DEFAULT 0,
    breadth_state   TEXT,
    ic_state        TEXT,
    meta            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_regime_transitions_date
    ON regime_transitions (as_of_date DESC);
