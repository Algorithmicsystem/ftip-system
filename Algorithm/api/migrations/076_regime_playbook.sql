-- Phase 12.2: Regime playbook
CREATE TABLE IF NOT EXISTS regime_playbook (
    regime_label                TEXT        NOT NULL PRIMARY KEY,
    recommended_signal_types    JSONB       DEFAULT '[]'::JSONB,
    avoided_signal_types        JSONB       DEFAULT '[]'::JSONB,
    factor_weights              JSONB       DEFAULT '{}'::JSONB,
    historical_accuracy         NUMERIC,
    sample_count                INTEGER,
    regime_duration_avg_days    NUMERIC,
    transition_probability      JSONB       DEFAULT '{}'::JSONB,
    last_updated                TIMESTAMPTZ DEFAULT now()
);
