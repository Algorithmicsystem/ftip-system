CREATE TABLE IF NOT EXISTS assistant_sessions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    title TEXT,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS assistant_messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT,
    tokens_in INT,
    tokens_out INT,
    extra JSONB
);

CREATE TABLE IF NOT EXISTS assistant_artifacts (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
