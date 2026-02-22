# Codex Fix Log

## Baseline Recon
- `pwd`: /Users/macuser/ftip-system/Algorithm
- `ls`: Makefile, README.md, api/, ftip/, tests/, requirements.txt, docker-compose.yml, etc.
- `git status -sb`: detached HEAD with existing local modifications in api/, ftip/, tests/.
- `git remote -v`: origin https://github.com/Algorithmicsystem/ftip-system.git
- `git branch --show-current`: (detached)
- `git log -1 --oneline`: 74fcc56 Merge pull request #73 from Algorithmicsystem/codex/fix-/providers/health-route-and-test

## Verification Attempt 1
- `pip install -r requirements.txt`
- `pytest -q`
- **Failure summary:** connection attempts to Postgres role `postgres` failed (DB unavailable), many DB-dependent tests failed.
- **Suspected cause:** local Postgres running on 5432 with user `macuser` (not `postgres`), plus missing migrations/routers.

## Verification Attempt 2 (DB URL override)
- `DATABASE_URL=postgresql://macuser@localhost:5432/postgres pytest -q`
- **Failures (13):**
  - Assistant chat missing API key returned logging error due to missing assistant tables + logging extra key `message`.
  - `/backtest/*`, `/signals/*`, and `/jobs/data/*` routes returned 404 (routers not included).
  - `/db/universe/load_default` 500 due to `source` column mismatch.
  - Migrations auto startup behavior didn’t respect `FTIP_MIGRATIONS_AUTO` and didn’t return applied list.
  - Market data tables and unique indexes missing (market_* migrations not registered).
  - `prosperity_signals_daily` PK missing `score_mode` (migration not registered).
  - `api.prosperity.query.coverage_for_universe` missing.
  - `db.upsert_universe` / `db.get_universe` missing.

## Fix Plan
- Register missing migrations (assistant tables, market_* tables, PK fix, backtest tables).
- Add missing `db.upsert_universe`, `db.get_universe`, and `coverage_for_universe` helpers.
- Include missing routers (backtest, signals, jobs: market_data/features/signals).
- Respect `FTIP_MIGRATIONS_AUTO` in startup; return applied list.
- Remove `source` column usage in `/db/universe/load_default` insert.
- Fix logging `extra` key collision and fail-fast on missing LLM API key.

## Verification Result
- Pending (re-run pytest after fixes).

## Verification Result (After Fix)
- `DATABASE_URL=postgresql://macuser@localhost:5432/postgres pytest -q`
- **PASS:** 106 passed, 4 skipped.

## Verification Attempt 3 (Post-push, env isolation + rate-limit fixes)
- `DATABASE_URL=postgresql://macuser@localhost:5432/postgres pytest -q`
- **PASS:** 116 passed, 4 skipped.
- `python -m ruff check .` -> PASS
- `python -m ruff format --check .` -> PASS

## Verification Result (Final)
- `DATABASE_URL=postgresql://macuser@localhost:5432/postgres pytest -q`
- **PASS:** 116 passed, 4 skipped.
- `python -m ruff check .` -> PASS
- `python -m ruff format --check .` -> PASS
