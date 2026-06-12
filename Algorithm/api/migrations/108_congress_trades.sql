CREATE TABLE IF NOT EXISTS congress_trades (
    id SERIAL PRIMARY KEY,
    member_name VARCHAR(200),
    chamber VARCHAR(10),
    symbol VARCHAR(20),
    trade_date DATE,
    transaction_type VARCHAR(10),
    amount_min BIGINT,
    amount_max BIGINT,
    disclosure_date DATE,
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS congress_trades_symbol_idx ON congress_trades(symbol);
CREATE INDEX IF NOT EXISTS congress_trades_date_idx ON congress_trades(trade_date DESC);
