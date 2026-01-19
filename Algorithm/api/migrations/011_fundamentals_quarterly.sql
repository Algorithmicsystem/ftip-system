CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    symbol TEXT NOT NULL REFERENCES market_symbols(symbol) ON DELETE CASCADE,
    fiscal_period_end DATE NOT NULL,
    report_date DATE NULL,
    revenue NUMERIC NULL,
    eps NUMERIC NULL,
    gross_margin NUMERIC NULL,
    op_margin NUMERIC NULL,
    fcf NUMERIC NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(symbol, fiscal_period_end)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_symbol_period ON fundamentals_quarterly(symbol, fiscal_period_end DESC);
