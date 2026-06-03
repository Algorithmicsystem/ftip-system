CREATE TABLE IF NOT EXISTS ml_signal_predictions (
    symbol                TEXT NOT NULL,
    as_of_date            DATE NOT NULL,
    model_version         TEXT NOT NULL,
    ml_prediction         INT,
    ml_confidence         NUMERIC,
    ml_agrees_with_axiom  BOOLEAN,
    regime_label          TEXT,
    feature_vector        JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_ml_signal_predictions_date
    ON ml_signal_predictions (as_of_date DESC, symbol);
