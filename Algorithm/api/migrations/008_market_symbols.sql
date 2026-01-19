CREATE TABLE IF NOT EXISTS market_symbols (
    symbol TEXT PRIMARY KEY,
    exchange TEXT NULL,
    country TEXT NULL,
    currency TEXT NULL,
    name TEXT NULL,
    sector TEXT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_market_symbols_active ON market_symbols(is_active);
CREATE INDEX IF NOT EXISTS idx_market_symbols_country ON market_symbols(country);
