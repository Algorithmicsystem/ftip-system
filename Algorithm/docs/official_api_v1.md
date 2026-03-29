# FTIP Official API v1 Contract

_Last updated: 2026-03-29._

This document defines the official FTIP API v1 path. For v1 integrations, use this flow only.

## Official v1 path (strict)

1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run`
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`
5. Optional scheduled equivalent: `POST /jobs/prosperity/daily-snapshot`

## Auth and headers

When any API key env var is set (`FTIP_API_KEY`, `FTIP_API_KEYS`, `FTIP_API_KEY_PRIMARY`), include one of:

- `X-FTIP-API-Key: <key>`
- `Authorization: Bearer <key>`

Use `Content-Type: application/json` on `POST` requests.

## Example end-to-end (official v1)

```bash
BASE=http://localhost:8000

curl -X POST "$BASE/prosperity/bootstrap"

curl -X POST "$BASE/prosperity/snapshot/run" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'

curl "$BASE/prosperity/latest/signal?symbol=AAPL&lookback=252"

curl "$BASE/prosperity/latest/features?symbol=AAPL&lookback=252"
```

Optional scheduler path:

```bash
curl -X POST "$BASE/jobs/prosperity/daily-snapshot" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL","MSFT"],"from_date":"2024-01-01","to_date":"2024-01-31","as_of_date":"2024-01-31","lookback":252}'
```

## Non-v1 surfaces (kept, not removed)

The following remain supported for compatibility and internal/operator workflows, but are not part of the official v1 product contract:

- `/signals/*` (legacy parallel signal namespace; prefer `/prosperity/latest/signal`)
- Assistant/Narrator/Backtest/Friction/Data families (`/assistant/*`, `/narrator/*`, `/backtest/*`, `/friction/*`, `/data/*`)
- Extended jobs/admin/diagnostic surfaces beyond `POST /jobs/prosperity/daily-snapshot`
- Prosperity operator endpoints outside the v1 sequence (for example ingest/control/coverage/health routes)

## Compatibility note

This is a presentation and onboarding contract update only. Non-v1 routes are not removed in Tier 0.
