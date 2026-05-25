CREATE TABLE IF NOT EXISTS market_breadth_daily (
    as_of_date               DATE PRIMARY KEY,
    breadth_confirmation_score       NUMERIC,
    participation_breadth_score      NUMERIC,
    breadth_thrust_proxy             NUMERIC,
    cross_sectional_dispersion_proxy NUMERIC,
    leadership_concentration_score   NUMERIC,
    internal_market_divergence_score NUMERIC,
    leader_strength_score            NUMERIC,
    laggard_pressure_score           NUMERIC,
    narrow_leadership_warning        BOOLEAN,
    broad_participation_confirmation BOOLEAN,
    breadth_state                    TEXT,
    universe_size                    INT,
    meta                             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_market_breadth_daily_date
    ON market_breadth_daily (as_of_date DESC);
