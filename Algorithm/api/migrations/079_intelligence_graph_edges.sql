-- Phase 12.4: Intelligence graph edges
CREATE TABLE IF NOT EXISTS intelligence_graph_edges (
    edge_id             TEXT        NOT NULL PRIMARY KEY,
    source_node_id      TEXT        NOT NULL,
    target_node_id      TEXT        NOT NULL,
    edge_type           TEXT        NOT NULL,
    weight              NUMERIC     DEFAULT 1.0,
    direction           TEXT        DEFAULT 'bidirectional',
    last_validated      DATE
);

CREATE INDEX IF NOT EXISTS idx_ige_source
    ON intelligence_graph_edges (source_node_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_ige_target
    ON intelligence_graph_edges (target_node_id, edge_type);
