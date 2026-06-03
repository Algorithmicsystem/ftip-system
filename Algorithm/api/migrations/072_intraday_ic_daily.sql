CREATE TABLE IF NOT EXISTS intraday_ic_daily (
    session_date DATE NOT NULL,
    update_hour  INT NOT NULL,
    ic_value     NUMERIC,
    sample_count INT NOT NULL DEFAULT 0,
    ic_state     TEXT NOT NULL DEFAULT 'INSUFFICIENT',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_date, update_hour)
);
