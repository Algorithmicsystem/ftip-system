CREATE TABLE IF NOT EXISTS data_versions (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_snapshot_hash TEXT NOT NULL,
    code_sha TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_name, source_snapshot_hash, code_sha)
);

CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    country TEXT,
    exchange TEXT,
    currency TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_symbols_country_exchange
ON symbols(country, exchange);

CREATE TABLE IF NOT EXISTS universe_membership (
    id BIGSERIAL PRIMARY KEY,
    universe_name TEXT NOT NULL,
    symbol TEXT NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    start_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    end_ts TIMESTAMPTZ,
    data_version_id BIGINT REFERENCES data_versions(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (universe_name, symbol, start_ts)
);

CREATE INDEX IF NOT EXISTS idx_universe_membership_lookup
ON universe_membership(universe_name, symbol, start_ts, end_ts);

CREATE TABLE IF NOT EXISTS prices_daily_versioned (
    id BIGSERIAL PRIMARY KEY,
    data_version_id BIGINT NOT NULL REFERENCES data_versions(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    date DATE NOT NULL,
    as_of_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    currency TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (data_version_id, symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_daily_symbol_date_asof
ON prices_daily_versioned(symbol, date, as_of_ts DESC);

CREATE TABLE IF NOT EXISTS corp_actions_versioned (
    id BIGSERIAL PRIMARY KEY,
    data_version_id BIGINT NOT NULL REFERENCES data_versions(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    effective_date DATE NOT NULL,
    factor DOUBLE PRECISION,
    value DOUBLE PRECISION,
    announced_ts TIMESTAMPTZ NOT NULL,
    as_of_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (data_version_id, symbol, action_type, effective_date, announced_ts)
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol_effective_asof
ON corp_actions_versioned(symbol, effective_date, as_of_ts DESC);

CREATE TABLE IF NOT EXISTS fundamentals_pit (
    id BIGSERIAL PRIMARY KEY,
    data_version_id BIGINT NOT NULL REFERENCES data_versions(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    metric_key TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    period_end DATE NOT NULL,
    published_ts TIMESTAMPTZ NOT NULL,
    as_of_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (data_version_id, symbol, metric_key, period_end, published_ts)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_symbol_asof
ON fundamentals_pit(symbol, as_of_ts DESC);

CREATE TABLE IF NOT EXISTS news_items (
    id BIGSERIAL PRIMARY KEY,
    data_version_id BIGINT NOT NULL REFERENCES data_versions(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    published_ts TIMESTAMPTZ NOT NULL,
    as_of_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL,
    credibility DOUBLE PRECISION,
    headline TEXT NOT NULL,
    headline_hash TEXT NOT NULL,
    full_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (data_version_id, symbol, published_ts, source, headline_hash)
);

CREATE INDEX IF NOT EXISTS idx_news_items_symbol_asof
ON news_items(symbol, as_of_ts DESC);
