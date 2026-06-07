-- Migration 097: ML signal predictions table for ensemble blending.
CREATE TABLE IF NOT EXISTS ml_signal_predictions (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    as_of_date      DATE NOT NULL,
    model_version   TEXT NOT NULL,
    prediction_score NUMERIC(8,6),
    signal_label    TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (symbol, as_of_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_ml_signal_predictions_sym_date
    ON ml_signal_predictions (symbol, as_of_date DESC);
