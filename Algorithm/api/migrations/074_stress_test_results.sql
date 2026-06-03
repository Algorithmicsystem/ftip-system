CREATE TABLE IF NOT EXISTS stress_test_results (
    portfolio_id              TEXT        NOT NULL,
    as_of_date                DATE        NOT NULL,
    scenario_label            TEXT        NOT NULL,
    portfolio_loss_pct        NUMERIC,
    worst_positions           JSONB,
    recovery_days_estimate    INT,
    meta                      JSONB,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (portfolio_id, as_of_date, scenario_label)
);
