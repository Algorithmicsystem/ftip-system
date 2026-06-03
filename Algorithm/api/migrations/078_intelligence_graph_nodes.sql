-- Phase 12.4: Intelligence graph nodes
CREATE TABLE IF NOT EXISTS intelligence_graph_nodes (
    node_id         TEXT        NOT NULL PRIMARY KEY,
    node_type       TEXT        NOT NULL,
    label           TEXT        NOT NULL,
    axiom_score     NUMERIC,
    regime_label    TEXT,
    sri_exposure    NUMERIC     DEFAULT 0.0,
    last_updated    DATE
);
