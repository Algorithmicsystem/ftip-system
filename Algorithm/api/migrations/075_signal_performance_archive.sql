-- Phase 12.1: Signal performance archive
CREATE TABLE IF NOT EXISTS signal_performance_archive (
    symbol                    TEXT        NOT NULL,
    signal_date               DATE        NOT NULL,
    signal_label              TEXT        NOT NULL DEFAULT 'BUY',
    dau_at_signal             NUMERIC,
    regime_at_signal          TEXT,
    primary_factor_driver     TEXT,
    horizon_5d_return         NUMERIC,
    horizon_21d_return        NUMERIC,
    batting_average           NUMERIC,
    slugging_average          NUMERIC,
    signal_war                NUMERIC,
    genealogy                 JSONB       DEFAULT '{}'::JSONB,
    computed_at               TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (symbol, signal_date)
);

CREATE INDEX IF NOT EXISTS idx_spa_regime
    ON signal_performance_archive (regime_at_signal, signal_date DESC);

CREATE INDEX IF NOT EXISTS idx_spa_war
    ON signal_performance_archive (signal_war DESC NULLS LAST);
