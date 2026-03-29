# FTIP Official v1 Runbook (DB-backed)

_Last updated: 2026-03-29._

This is the **single operator/developer path** for official v1.

## What “official v1” is

Official v1 means this exact runtime flow only:

1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run`
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`
5. Optional scheduler equivalent: `POST /jobs/prosperity/daily-snapshot`

If you are integrating FTIP v1, use this flow and treat other routes as out-of-scope for product signoff.

## Local startup (strict DB-backed v1)

From `Algorithm/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start Postgres (choose one):

```bash
docker-compose up -d db
# or: brew services start postgresql@16
```

Set strict v1 env:

```bash
export FTIP_DB_ENABLED=1
export FTIP_DB_READ_ENABLED=1
export FTIP_DB_WRITE_ENABLED=1
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ftip
export FTIP_DB_REQUIRED=1
export FTIP_MIGRATIONS_AUTO=1
```

Start API:

```bash
make dev
```

## DB-backed signoff (minimum)

In a second terminal:

```bash
BASE=http://localhost:8000

curl -fsS -X POST "$BASE/prosperity/bootstrap"

curl -fsS -X POST "$BASE/prosperity/snapshot/run" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'

curl -fsS "$BASE/prosperity/latest/signal?symbol=AAPL&lookback=252"

curl -fsS "$BASE/prosperity/latest/features?symbol=AAPL&lookback=252"
```

Expected signoff outcome:

- all four calls return HTTP `200`
- `latest/signal` returns one canonical latest record for the query key
- `latest/features` returns the matching latest feature payload

Optional scheduler parity check:

```bash
curl -fsS -X POST "$BASE/jobs/prosperity/daily-snapshot" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
```

## Deployment-safe env expectations

Required for production v1:

- `FTIP_DB_ENABLED=1`
- `FTIP_DB_READ_ENABLED=1`
- `FTIP_DB_WRITE_ENABLED=1`
- `DATABASE_URL=postgresql://...`
- `FTIP_DB_REQUIRED=1`

Migration/bootstrap policy (pick exactly one and document it in your deploy config):

- **Auto migration at startup**: `FTIP_MIGRATIONS_AUTO=1`
- **Manual migration/bootstrap before traffic**: `FTIP_MIGRATIONS_AUTO=0` and run `POST /prosperity/bootstrap` successfully before routing traffic

Safety behavior with `FTIP_DB_REQUIRED=1`:

- service fails fast if DB flags are inconsistent
- service fails fast if DB is unreachable
- service fails fast if required v1 tables are missing when auto-migrations are disabled

## Intentionally out of scope for official v1

Not part of official v1 signoff (may exist for compatibility/internal use):

- `/signals/*` legacy signal namespace
- assistant/narrator/backtest/friction/data surfaces
- non-v1 prosperity operator/diagnostic routes beyond the sequence above
- broad platform certification outside DB-backed prosperity v1 flow

Keep these out of v1 acceptance criteria.
