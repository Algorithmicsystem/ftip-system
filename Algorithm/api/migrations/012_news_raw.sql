CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS news_raw (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    url_hash TEXT NOT NULL UNIQUE,
    content_snippet TEXT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_news_raw_symbol_published ON news_raw(symbol, published_at DESC);
