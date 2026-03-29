# FTIP Official API v1 Contract

_Last updated: 2026-03-29._

This document defines the official FTIP API v1 contract. For local run, DB-backed signoff, and deployment-safe operations, follow `docs/official_v1_runbook.md`.

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

## Runtime contract (DB-backed v1)

The official v1 contract assumes DB-backed runtime:

- `FTIP_DB_ENABLED=1`
- `FTIP_DB_WRITE_ENABLED=1`
- `FTIP_DB_READ_ENABLED=1`
- `DATABASE_URL=postgresql://...`

Safe deployment default:

- `FTIP_DB_REQUIRED=1`

Migration/bootstrap policy:

- either `FTIP_MIGRATIONS_AUTO=1`, or
- `FTIP_MIGRATIONS_AUTO=0` and run `POST /prosperity/bootstrap` successfully before serving traffic

With `FTIP_DB_REQUIRED=1`, startup is fail-fast for inconsistent DB flags, unreachable DB, or missing required v1 tables when auto migrations are disabled.

## Out of scope for official v1

These may remain available but are not part of the official v1 product contract:

- `/signals/*` and other legacy/parallel signal surfaces
- assistant/narrator/backtest/friction/data families
- non-v1 prosperity operator/admin/diagnostic endpoints

## Compatibility note

This is a contract/surface definition. Non-v1 routes are not removed by this document.
