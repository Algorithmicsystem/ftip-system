-- Migration 107: AI Document Extraction Audit Table
-- Stores every extraction attempt with confidence scores and review queue.

CREATE TABLE IF NOT EXISTS extraction_audit (
    id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id            TEXT,
    entity_type          TEXT,                           -- 'pe_portco' | 'smb_entity'
    filename             TEXT        NOT NULL,
    document_type        TEXT,
    period               TEXT,
    period_end_date      DATE,
    overall_confidence   NUMERIC,
    extracted_json       JSONB,                          -- full ExtractionResult as JSON
    fields_needing_review TEXT[],
    status               TEXT        NOT NULL DEFAULT 'pending',  -- 'pending'|'approved'|'rejected'
    reviewed_by          TEXT,
    reviewed_at          TIMESTAMPTZ,
    corrections          JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_extraction_audit_entity
    ON extraction_audit(entity_id, entity_type);

CREATE INDEX IF NOT EXISTS idx_extraction_audit_status
    ON extraction_audit(status) WHERE status = 'pending';

COMMENT ON TABLE extraction_audit IS
    'Audit log for AI financial document extractions. '
    'Pending rows are surfaced in the review queue for human correction.';
