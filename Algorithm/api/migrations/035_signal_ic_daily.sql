CREATE TABLE IF NOT EXISTS signal_ic_daily (
    as_of_date      DATE        NOT NULL,
    score_field     TEXT        NOT NULL,   -- 'composite', 'fundamental_reality', etc.
    horizon_label   TEXT        NOT NULL,   -- '5d', '21d', '63d'
    ic_value        NUMERIC,               -- Spearman rank correlation for this date
    sample_size     INT,                   -- number of symbols used
    p_value         NUMERIC,               -- two-tailed significance
    ic_mean_21d     NUMERIC,               -- rolling 21-period mean IC
    ic_mean_63d     NUMERIC,               -- rolling 63-period mean IC
    icir_21d        NUMERIC,               -- IC information ratio over 21 periods
    icir_63d        NUMERIC,               -- IC information ratio over 63 periods
    t_stat          NUMERIC,               -- single-sample t-stat for this IC value
    ic_state        TEXT,                  -- 'STRONG','MODERATE','WEAK','DEGRADED','INSUFFICIENT'
    meta            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (as_of_date, score_field, horizon_label)
);

CREATE INDEX IF NOT EXISTS idx_signal_ic_daily_date
    ON signal_ic_daily (as_of_date DESC, score_field, horizon_label);

CREATE INDEX IF NOT EXISTS idx_signal_ic_daily_field_horizon
    ON signal_ic_daily (score_field, horizon_label, as_of_date DESC);
