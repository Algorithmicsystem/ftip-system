# FTIP System Notes

## DB smoke tests

Set the environment for database access (for local testing you can point `DATABASE_URL` at a Postgres instance) and enable DB endpoints:

```bash
export FTIP_DB_ENABLED=1
export DATABASE_URL=postgresql://user:pass@host:5432/ftip
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

## Phase 3 narrator verification

Set `FTIP_API_KEY` (or `FTIP_API_KEYS`) and `OPENAI_API_KEY` on Railway, then run the official checklist to validate the narrator endpoints and auth (requires `BASE` and `KEY`, using the `X-FTIP-API-Key` header):

```bash
BASE="https://ftip-system-production.up.railway.app" KEY="your-api-key" ./scripts/phase3_verify.sh
```

The script confirms `/health` and `/version` remain public, `/narrator/health` is locked down without the key, and `/narrator/ask` + `/narrator/explain-signal` return narrated responses when the OpenAI API key is configured.

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
