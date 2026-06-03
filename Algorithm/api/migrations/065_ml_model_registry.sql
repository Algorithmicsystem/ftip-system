CREATE TABLE IF NOT EXISTS ml_model_registry (
    model_id      TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    regime_label  TEXT,
    model_path    TEXT NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    trained_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    sample_count  INT,
    test_accuracy NUMERIC,
    test_roc_auc  NUMERIC,
    psi_score     NUMERIC,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ml_model_registry_active
    ON ml_model_registry (is_active, regime_label, trained_at DESC);
