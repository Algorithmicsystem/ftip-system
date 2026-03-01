# FTIP System Notes

## Local Development Runbook (zero-issues)

Follow these steps from a clean checkout (use **Terminal 1** unless noted).

1) `cd Algorithm`
2) `python3 -m venv .venv`
3) `source .venv/bin/activate`
4) `pip install -r requirements.txt`
5) `cp .env.example .env`
6) Choose **one** database option:
   - **Homebrew Postgres**:
     - `brew services start postgresql@16`
     - Update `.env` if needed (e.g. `DATABASE_URL=postgresql://$USER@localhost:5432/postgres`)
   - **Docker Compose v1** (only `docker-compose` is supported here):
     - `docker-compose up -d db`
7) `make migrate` (uses the internal migration runner in `api/migrations`, not Alembic)
8) `make db-check`
9) `make dev` (keep this running in **Terminal 1**)
10) In **Terminal 2**, verify endpoints:
    - `curl http://localhost:8000/health`
    - `curl http://localhost:8000/docs`
    - `curl http://localhost:8000/openapi.json`
11) `make test`

### Make targets (recommended)

```bash
make dev             # start the API (uvicorn)
make compose-up-db   # start Postgres via docker-compose (v1)
make compose-down-db # stop Postgres and remove volumes
make migrate         # run migrations only
make db-check        # verify database connectivity
make db-reset        # drop/recreate schema and re-run migrations
make env-check       # print FTIP_DB_ENABLED and DATABASE_URL
make test            # pytest -q
make lint            # ruff check .
make fmt             # ruff check --fix . && black .
```

### Troubleshooting

- **`zsh: command not found: python`**: run `python3 -m venv .venv` and `source .venv/bin/activate` (or use `.venv/bin/python` directly).
- **`ModuleNotFoundError: No module named 'psycopg'`**: make sure you installed deps inside the venv (`pip install -r requirements.txt`) and are using `.venv/bin/python`.
- **`curl: (7) Failed to connect` / connection refused**: the API isn't running; keep `make dev` running in Terminal 1.
- **`docker-compose.yml not found`**: run docker-compose from the `Algorithm` folder (where `docker-compose.yml` lives).

## Reset DB (safe for local/dev)

If you are running with Docker, the fastest reset is to drop the volume:

```bash
docker-compose down -v
docker-compose up -d db
make migrate
```

If you are running against a local Postgres instance, use the reset helper:

```bash
export FTIP_DB_ENABLED=1
export FTIP_MIGRATIONS_AUTO=1
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ftip
python tools/reset_db.py
```

## DB Health Check

```bash
BASE=http://localhost:8000 ./scripts/db_health_check.sh
```

## Migrations

This repo uses the internal migrations runner in `api/migrations` (not Alembic). To run migrations explicitly:

```bash
make migrate
```

Or call the legacy shim directly:

```bash
python -c "from api.migrations.runner import apply_migrations; apply_migrations()"
```

## Run Tests

```bash
cd Algorithm
pytest -q
```

## Deploy (Railway)

1. Create a Railway service from this repo.
2. Add the required environment variables (see `.env.example` and the production guardrails below).
3. Ensure the service start command uses the default `CMD` in the Dockerfile (runs `scripts/railway_start.sh`).
4. Configure a Postgres database in Railway and set `DATABASE_URL` accordingly.

## CHANGELOG

### Unreleased
- Added schema version reporting to `/db/health` to surface migration state and aid diagnostics.
- Added `.env.example` and helper scripts for DB health checks and resets.

## DB smoke tests

Set the environment for database access (for local testing you can point `DATABASE_URL` at a Postgres instance) and enable DB endpoints:

```bash
export FTIP_DB_ENABLED=1
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ftip
```

Then run a few smoke checks against a running server (default port 8000):

```bash
curl http://localhost:8000/db/health
curl -X POST http://localhost:8000/db/universe/load_default
curl -X POST http://localhost:8000/db/run_snapshot \
  -H "Content-Type: application/json" \
  -d '{"as_of":"2024-01-31","lookback":252,"active_only":true,"limit":25}'
curl -X POST http://localhost:8000/db/save_signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","as_of":"2024-01-31","lookback":252}'
```

Each call should return HTTP 200 with JSON payloads when the database is reachable.

## Strategy Graph Engine v1

The Strategy Graph Engine computes multi-strategy signals, ensembles them, and records audit metadata.

Run a daily computation across symbols:

```bash
curl -X POST http://localhost:8000/prosperity/strategy_graph/run \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
```

Fetch the latest ensemble or per-strategy breakdown:

```bash
curl "http://localhost:8000/prosperity/strategy_graph/latest/ensemble?symbol=AAPL&lookback=252"
curl "http://localhost:8000/prosperity/strategy_graph/latest/strategies?symbol=AAPL&lookback=252"
```

To smoke test locally, run:

```bash
bash tools/smoke_strategy_graph.sh
```

## Copy/paste safe zsh commands

When using zsh, enable `setopt interactivecomments` before copy/pasting any of the sample commands so inline `#` comments are ignored correctly:

```bash
setopt interactivecomments
```

If you see `command not found: #`, re-run the command above and try again.

## Prosperity DB v1

Prosperity DB adds a durable research warehouse (Postgres) for caching market data, features, and signals. Enable it with:

```bash
export FTIP_DB_ENABLED=1
export DATABASE_URL=postgresql://user:pass@host:5432/ftip
export FTIP_DB_WRITE_ENABLED=1
export FTIP_DB_READ_ENABLED=1
export FTIP_MIGRATIONS_AUTO=1
```

Example calls:

```bash
curl http://localhost:8000/prosperity/health
curl -X POST http://localhost:8000/prosperity/snapshot/run \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT","GOOGL","AMZN","NVDA"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
curl "http://localhost:8000/prosperity/latest/signal?symbol=AAPL&lookback=252"
```

To smoke test the snapshot->latest round trip on a fresh database, run:

```bash
curl -X POST http://localhost:8000/prosperity/bootstrap
curl -X POST http://localhost:8000/prosperity/snapshot/run \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL"],"from_date":"2024-01-01","to_date":"2024-01-05","as_of_date":"2024-01-05","lookback":252}'
curl "http://localhost:8000/prosperity/latest/signal?symbol=AAPL&lookback=252"
```

Each call should return HTTP 200 with the latest signal containing `score_mode`.

Prosperity endpoints require an API key when any of the following are set (merged + trimmed in this order):

- `FTIP_API_KEY`
- `FTIP_API_KEYS` (comma-separated)
- `FTIP_API_KEY_PRIMARY`

If no keys are provided, auth is disabled for local development. When keys are set, include the header `X-FTIP-API-Key` (or `Authorization: Bearer <key>`) on every `/prosperity/*` request. Check the status safely with `/auth/status` (requires a key unless `FTIP_AUTH_STATUS_PUBLIC=1`).

```bash
export FTIP_API_KEY="demo-key"
export BASE=http://localhost:8000
export KEY=demo-key

bash scripts/phase2_verify.sh
```

Production verification (after setting Railway variables):

```
BASE="https://ftip-system-production.up.railway.app" KEY="cfotwin-dev-2025-12-29" ./scripts/phase2_verify.sh
```

## Milestone E/F quickstart

- UI: `http://localhost:8000/app`
- Docs: `docs/runbook.md`
- Smoke script:

```bash
bash scripts/smoke_milestone_ef.sh
```

## Phase 3 narrator verification

Set `FTIP_API_KEY` (or `FTIP_API_KEYS`) and `OPENAI_API_KEY` on Railway, then run the official checklist to validate the narrator endpoints and auth (requires `BASE` and `KEY`, using the `X-FTIP-API-Key` header):

```bash
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/phase3_verify.sh
```

The script confirms `/health` and `/version` remain public, `/narrator/health` is locked down without the key, `/narrator/ask` returns a narrated response when the OpenAI API key is configured, and `/narrator/explain-signal` returns a deterministic explanation from evidence payloads.

To run a local end-to-end smoke check that exercises the prosperity endpoints:

```bash
bash tools/smoke_prosperity.sh
```

## Phase 4 verification (Narrator + Prosperity)

Phase 4 hardens the narrator endpoints and adds a deep integration with the prosperity strategy graph. After setting `FTIP_API_KEY` (or `FTIP_API_KEYS`) and `OPENAI_API_KEY`, run the official script to validate Railway or local deployments:

```bash
# Local example
BASE="http://localhost:8000" KEY="demo-key" ./scripts/phase4_verify.sh

# Production example (Railway)
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/phase4_verify.sh
```

The script confirms:

- `/health` and `/version` stay public.
- `/narrator/health` rejects missing API keys.
- `/prosperity/snapshot/run` can compute five symbols and writes both `signals` and `features` rows (using a 2023-01-01 → 2024-12-31 range to satisfy the 252-day lookback).
- `/narrator/explain-strategy-graph` returns a narrated graph with nodes.
- `/narrator/diagnose` reports a clean bill of health for auth, DB, migrations, and latest signals.

## Phase 5 verification (Narrator portfolio)

Phase 5 hardens the narrator portfolio endpoint so it always returns valid JSON, even when a backtest is skipped. After setting `FTIP_API_KEY` (or `FTIP_API_KEYS`) and `OPENAI_API_KEY`, run the official script to validate Railway or local deployments:

```bash
# Local example
BASE="http://localhost:8000" KEY="demo-key" ./scripts/phase5_verify.sh

# Production example (Railway)
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/phase5_verify.sh
```

The script confirms:

- `/health` and `/version` stay public.
- `/narrator/health` rejects missing API keys and succeeds with the key.
- `/prosperity/snapshot/run` can compute enough bars for a 252-day lookback and writes at least one `signals` row.
- `/narrator/portfolio` returns valid numeric performance fields when `include_backtest` is both `false` and `true`.

## Phase 6 verification (Production snapshot job)

Set the Railway variables to control the daily snapshot job runner:

```
FTIP_DB_ENABLED=1
FTIP_DB_WRITE_ENABLED=1
FTIP_DB_READ_ENABLED=1
FTIP_MIGRATIONS_AUTO=1
FTIP_API_KEY=your-api-key
FTIP_UNIVERSE="AAPL,MSFT,NVDA,AMZN,TSLA,GOOGL,META,JPM,XOM,BRK.B"
FTIP_LOOKBACK=252
FTIP_SNAPSHOT_WINDOW_DAYS=365
FTIP_SNAPSHOT_CONCURRENCY=3
# Optional: prune old rows
FTIP_RETENTION_DAYS=730
```

Trigger the protected job endpoint manually:

```bash
curl -X POST "${BASE:-http://localhost:8000}/jobs/prosperity/daily-snapshot" \
  -H "X-FTIP-API-Key: ${KEY}"
```

Then fetch the latest artifacts:

```bash
curl "${BASE:-http://localhost:8000}/prosperity/latest/signal?symbol=AAPL&lookback=252" -H "X-FTIP-API-Key: ${KEY}"
curl "${BASE:-http://localhost:8000}/prosperity/graph/strategy?symbol=AAPL&lookback=252&days=365" -H "X-FTIP-API-Key: ${KEY}"
```

To validate production (or local) deployments end-to-end, run:

```bash
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/phase6_verify.sh
```

## Phase 7 verification (Railway Cron + job status)

Set the Railway variables to control the automated daily snapshot job:

```
FTIP_UNIVERSE
FTIP_LOOKBACK
FTIP_SNAPSHOT_WINDOW_DAYS
FTIP_SNAPSHOT_CONCURRENCY
FTIP_RETENTION_DAYS (optional)
FTIP_JOB_LOCK_TTL_SECONDS (optional)
FTIP_DB_ENABLED=1
FTIP_DB_WRITE_ENABLED=1
FTIP_DB_READ_ENABLED=1
FTIP_MIGRATIONS_AUTO=1
FTIP_API_KEY=your-api-key
```

Railway Cron should call the protected job endpoint with POST to trigger the snapshot run:

```
curl -s -X POST "$BASE/jobs/prosperity/daily-snapshot" -H "X-FTIP-API-Key: $KEY"
```

If a run is already in progress, an immediate second POST will return HTTP 409 with a JSON error body indicating the lock state.

Check the job status and overlap lock using the official verification script:

```bash
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/phase7_verify.sh
```

## Milestone A verification (Automation + Reliability)

Run the Milestone A verification script after deploying the automation and reliability updates:

```bash
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/milestoneA_verify.sh
```

## Assistant / Narrator (Phase 5)

Environment variables:

```bash
export FTIP_LLM_ENABLED=1          # enable the assistant endpoints
export OPENAI_API_KEY=sk-...       # required when LLM is enabled
export FTIP_MIGRATIONS_AUTO=1      # run migrations automatically at startup
```

Health check:

```bash
curl http://localhost:8000/assistant/health
```

Chat example (persists a session when FTIP_DB_ENABLED=1):

```bash
curl -X POST http://localhost:8000/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How was the signal computed for AAPL?"}'
```

Explain a signal (reuses the internal /signal logic):

```bash
curl -X POST http://localhost:8000/assistant/explain/signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","as_of":"2024-01-31","lookback":252}'
```

The narrator is safety-first and should never provide financial advice. It explains what the model computed, which thresholds were used, and what calibration metadata was present.

## FTIP Narrator (OpenAI)

Environment variables (LLM feature-flagged by default):

```bash
export FTIP_LLM_ENABLED=1               # enable the narrator endpoints (default is 0)
export OPENAI_API_KEY=sk-...            # required (legacy OpenAI_ftip-system still supported)
export FTIP_LLM_MODEL=gpt-4o-mini       # optional override
export FTIP_LLM_MAX_TOKENS=700          # optional override
export FTIP_LLM_TEMPERATURE=0.2         # optional override
```

Health check:

```bash
curl "http://localhost:8000/prosperity/narrator/health"
```

Explain a symbol using DB-grounded context (strategy graph ensemble + strategies + features + latest signal):

```bash
curl "http://localhost:8000/prosperity/narrator/explain?symbol=AAPL&as_of_date=2024-12-31&lookback=252"
```

Ask system or comparative symbol questions (returns 404 if DB context is missing):

```bash
curl -X POST http://localhost:8000/prosperity/narrator/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Why is AAPL a BUY?","symbols":["AAPL","MSFT"],"as_of_date":"2024-12-31","lookback":252}'
```

Recommended flow: bootstrap the DB, run `/prosperity/strategy_graph/run` to populate ensembles, then call `/prosperity/narrator/explain` for narration. The narrator refuses to answer when DB context is missing and always returns a trace ID for auditing.
## Phase 3 Production Hardening

Set the new production guardrails before shipping:

```bash
export FTIP_API_KEYS="prod-api-key-1,prod-api-key-2"   # or FTIP_API_KEYS_JSON='["key1","key2"]'
export FTIP_PUBLIC_DOCS=0                               # set to 1 to expose /docs + /openapi.json without a key
export FTIP_ALLOWED_ORIGINS="https://cfotwin.ca,https://www.cfotwin.ca"  # CORS allow-list
export FTIP_RATE_LIMIT_RPM=120                          # requests per minute per API key/IP
```

On Railway, set the custom domain under **Settings → Domains** to `cfotwin.ca` (and `www.cfotwin.ca` if you want the www alias)
after adding the DNS records in Cloudflare. No secrets should be hardcoded; rely on Railway environment variables.

Copy/paste friendly curl examples (include the required API key header for prosperity endpoints):

```bash
API_KEY=prod-api-key-1
BASE="https://cfotwin.ca"  # or your preview URL

curl -H "X-FTIP-API-Key: ${API_KEY}" "${BASE}/prosperity/narrator/health"
curl -H "X-FTIP-API-Key: ${API_KEY}" "${BASE}/prosperity/strategy_graph/latest/ensemble?symbol=AAPL&lookback=252"
curl -H "X-FTIP-API-Key: ${API_KEY}" -X POST "${BASE}/prosperity/snapshot/run" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
```

Operational endpoints (no secrets) for observability:

```bash
curl "${BASE}/ops/domain"     # shows allowed origins + base URL
curl "${BASE}/ops/metrics"    # JSON counters (requests, rate-limit hits, narrator calls, etc.)
curl "${BASE}/ops/last_runs"  # recent snapshot/strategy_graph executions
```

To run the phase 3 verifier locally against a running server:

```bash
BASE_URL=http://localhost:8000 API_KEY=${API_KEY} bash scripts/verify_phase3.sh
```

## Milestone B+C+D (Data Backbone + Feature Engine + Signal Engine)

Canonical symbol formats:

- US tickers: `AAPL`, `NVDA`.
- Canada tickers: `SHOP.TO` (TSX) or `PNG.V` (TSXV).

Free-first data adapters use Stooq for US daily bars and YFinance for intraday (when installed). If intraday data is unavailable, the intraday job still returns HTTP 200 with `status=partial` and a typed `reason_code` such as `PROVIDER_UNAVAILABLE`.

### Protected job endpoints

All `/jobs/*` routes require `X-FTIP-API-Key` when auth is enabled:

- `POST /jobs/data/universe` (upsert symbol universe, `mode=default` for the US+CA starter set)
- `POST /jobs/data/bars-daily`
- `POST /jobs/data/bars-intraday`
- `POST /jobs/data/news`
- `POST /jobs/data/fundamentals`
- `POST /jobs/data/sentiment-daily`
- `POST /jobs/features/daily`
- `POST /jobs/features/intraday`
- `POST /jobs/signals/daily`
- `POST /jobs/signals/intraday`

### Protected query endpoints

- `GET /signals/latest?symbol=AAPL`
- `GET /signals/top?mode=buy&limit=10&country=US`
- `GET /signals/evidence?symbol=AAPL&as_of_date=YYYY-MM-DD`

### Narrator evidence endpoint

`POST /narrator/explain-signal` accepts the evidence payload returned by `/signals/evidence` and returns a deterministic explanation, bullet reasons, and risk notes without invoking an LLM.

### Verification script

Run the end-to-end Milestone B+C+D checks (uses POSIX sh + python3 helpers):

```bash
chmod +x scripts/milestoneB_verify.sh
BASE="http://localhost:8000" KEY="demo-key" ./scripts/milestoneB_verify.sh
```

## Phase 1: Versioned Reality Quickstart (US + Canada, daily PIT)

This phase adds a replayable point-in-time (PIT) data layer with strict no-lookahead semantics.

### Environment

```bash
cd Algorithm
source .venv/bin/activate
export FTIP_DB_ENABLED=1
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
export FTIP_MIGRATIONS_AUTO=1
```

Run migrations:

```bash
python -c "from api.migrations.runner import apply_migrations; apply_migrations()"
```

### Create a data version

```bash
curl -X POST http://localhost:8000/data/version/create \
  -H "Content-Type: application/json" \
  -d '{"source_name":"manual","source_snapshot_hash":"snap-2026-01-01","notes":"daily load"}'
```

### Ingest sample daily prices

```bash
curl -X POST http://localhost:8000/data/prices/ingest_daily \
  -H "Content-Type: application/json" \
  -d '{"data_version_id":1,"items":[{"symbol":"AAPL","date":"2024-05-01","open":170,"high":172,"low":169,"close":171.5,"volume":1000000}]}'
```

### Ingest sample fundamentals (PIT)

```bash
curl -X POST http://localhost:8000/data/fundamentals/ingest_pit \
  -H "Content-Type: application/json" \
  -d '{"data_version_id":1,"items":[{"symbol":"AAPL","metric_key":"EPS","metric_value":1.9,"period_end":"2024-03-31","published_ts":"2024-04-30T00:00:00Z"}]}'
```

### Ingest sample news (PIT)

```bash
curl -X POST http://localhost:8000/data/news/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_version_id":1,"items":[{"symbol":"AAPL","published_ts":"2024-05-01T13:00:00Z","source":"wire","credibility":0.88,"headline":"Sample headline"}]}'
```

### Query as-of (time travel)

```bash
curl "http://localhost:8000/data/prices/query_daily?symbol=AAPL&start_date=2024-05-01&end_date=2024-05-01&as_of_ts=2024-05-02T00:00:00Z"
curl "http://localhost:8000/data/fundamentals/query_pit?symbol=AAPL&as_of_ts=2024-05-15T00:00:00Z&metric_keys=EPS"
curl "http://localhost:8000/data/news/query?symbol=AAPL&as_of_ts=2024-05-15T00:00:00Z&limit=50"
```

No lookahead is enforced in PIT query helpers: fundamentals and news are filtered by `published_ts <= as_of_ts` and `as_of_ts <= query_as_of_ts`.

## Phase 2: Market Microstructure + Trading Friction

Phase 2 adds a modular daily-bar friction engine under `ftip/friction/` with pluggable slippage, spread, impact, fill, cost, and constraints components. It is deterministic under fixed seed and supports US/Canada daily strategies only (no intraday execution).

### Friction simulation endpoint

Use the new debugging endpoint:

```bash
curl -X POST http://localhost:8000/friction/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "cost_model": {"fee_bps": 1, "slippage_bps": 5, "seed": 42},
    "market_state": {
      "date": "2024-01-02", "open": 100, "high": 102, "low": 99,
      "close": 101, "volume": 1000000
    },
    "execution_plan": {
      "symbol": "AAPL", "date": "2024-01-02", "side": "BUY",
      "notional": 10000, "order_type": "MARKET"
    }
  }'
```

### Backtesting with friction enabled

Existing `/backtest/run` accepts old `cost_model` payloads (`fee_bps`, `slippage_bps`) and now also supports enhanced friction fields (spread, impact, ADV/participation, overnight gap penalty, limit order controls, and seed). Backtest costs are now executed via the friction engine.

> Note: execution model is daily-only and does not support intraday routing/fills.
