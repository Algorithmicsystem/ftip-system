-- Per-sector breadth metrics: one row per (as_of_date, sector)
CREATE TABLE IF NOT EXISTS sector_breadth_daily (
    as_of_date                  DATE        NOT NULL,
    sector                      TEXT        NOT NULL,
    symbol_count                INT         NOT NULL DEFAULT 0,
    buy_count                   INT         NOT NULL DEFAULT 0,
    sell_count                  INT         NOT NULL DEFAULT 0,
    hold_count                  INT         NOT NULL DEFAULT 0,
    avg_score                   NUMERIC,
    breadth_confirmation_score  NUMERIC,
    participation_breadth_score NUMERIC,
    breadth_state               TEXT,
    rotation_rank               INT,
    meta                        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (as_of_date, sector)
);

CREATE INDEX IF NOT EXISTS idx_sector_breadth_date
    ON sector_breadth_daily (as_of_date DESC, sector);
