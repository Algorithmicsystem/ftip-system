CREATE TABLE IF NOT EXISTS morning_briefings (
    date        DATE PRIMARY KEY,
    briefing    JSONB NOT NULL DEFAULT '{}'::jsonb,
    sri         NUMERIC,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
