CREATE INDEX IF NOT EXISTS idx_assistant_messages_session_created ON assistant_messages (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_assistant_artifacts_session_created ON assistant_artifacts (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_assistant_sessions_updated ON assistant_sessions (updated_at DESC);
