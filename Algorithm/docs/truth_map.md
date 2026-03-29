# FTIP System Truth Map (Code Audit)

## Verified Working Flows

1. **Prosperity snapshot pipeline (DB-enabled) is a real end-to-end path**.
   - `/prosperity/snapshot/run` normalizes symbols, ingests bars, validates minimum bars, computes features + signal payloads, persists outputs, tracks coverage, and returns per-symbol success/failure with row counts.
   - `/prosperity/latest/signal` and `/prosperity/latest/features` read back persisted results.
   - This route is also the path exercised by the repository smoke script.

2. **Scheduled daily prosperity snapshot job is real and wired to the snapshot pipeline**.
   - `/jobs/prosperity/daily-snapshot` acquires DB job locks, constructs a `SnapshotRunRequest`, calls `snapshot_run`, writes job status, and applies optional retention cleanup.

3. **Strategy Graph computation/persistence is real**.
   - `/prosperity/strategy_graph/run` fetches/loads candles, computes strategy outputs and ensemble via `ftip.strategy_graph.compute_strategy_graph`, then persists strategy + ensemble rows when DB write is enabled.
   - `/prosperity/strategy_graph/latest/ensemble` and `/latest/strategies` return persisted latest outputs.

4. **Jobs-based market-data/features/signals pipeline is real (DB path)**.
   - `/jobs/market_data/*` ingests bars/news/fundamentals/sentiment.
   - `/jobs/features/daily` computes and stores feature rows + quality score updates.
   - `/jobs/signals/daily` reads features and writes signals.

5. **Backtest API is real (not placeholder), including friction engine integration**.
   - `/backtest/run` executes `api.backtest.service.run_backtest`, pulls bars/signals/features from DB, applies friction model, writes backtest run/results/equity rows, and exposes retrieval endpoints.

## Partial or Broken Flows

1. **Most “end-to-end” test coverage for assistant/backtest jobs is heavily mocked**.
   - `test_assistant_orchestrator_endpoints.py` monkeypatches freshness/features/signals fetchers, so it verifies response schema, not true runtime integration.
   - `test_backtest_endpoints.py` monkeypatches `backtest.service` completely; route wiring is tested, not engine correctness.

2. **Prosperity route fallbacks can bypass DB and use live fetch fallback logic, creating mixed execution modes**.
   - In strategy graph route, if DB bars are absent, code attempts `massive_fetch_daily_bars` fallback.
   - This makes behavior environment-dependent (DB + provider availability), so production reliability depends on external connectivity/provider behavior.

3. **Signal quality controls are partially implemented**.
   - Risk/correlation overlay path calls `correlation_guard_stub`, which currently returns weights unchanged.

## Placeholder / Stub / Synthetic Logic

1. **Explicit legacy stubs remain in jobs code**.
   - `_release_job_lock` and `_insert_job_run` in `api/jobs/prosperity.py` are documented as “Legacy stub retained for backwards compatibility” and return `None`.

2. **Correlation guard is explicitly a stub/no-op**.
   - `ftip/risk.py::correlation_guard_stub(...)` ignores inputs and returns the input weights.

3. **Prosperity schema helper is placeholder-only**.
   - `api/prosperity/schema.py` states it is a placeholder for future extensions and currently only wraps `json.dumps`.

4. **Legacy/simple backtest function in `api.main` is explicitly marked placeholder**.
   - `run_backtest_core` is commented as a simple buy-and-hold placeholder for `/run_backtest` (distinct from the newer `/backtest/*` service).

5. **Web app contains explicit placeholder panels**.
   - `api/webapp/index.html` includes “Chart panel (placeholder)”, “Sentiment panel placeholder”, and “Fundamentals panel placeholder”.

## Main Product Path

**Single most coherent v1 flow in current code: Prosperity DB pipeline + Strategy Graph readouts.**

Recommended official v1 user-facing sequence:
1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run` (or scheduler: `POST /jobs/prosperity/daily-snapshot`)
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`
5. Optional explainability overlays:
   - `GET /prosperity/coverage`
   - `GET /prosperity/strategy_graph/latest/ensemble`
   - `GET /prosperity/strategy_graph/latest/strategies`

Why this is the best v1 path:
- It has the strongest route/service/storage continuity.
- It has lock/retention/coverage machinery.
- It is backed by explicit smoke scripts and multiple tests focused on these endpoints.

## Non-Core or Legacy Areas

1. **Two narrator stacks exist (overlap/duplication)**.
   - `api/narrator/routes.py` provides `/narrator/*` endpoints (`ask`, `explain-signal`, `explain-strategy-graph`, etc.).
   - `api/llm/routes.py` also provides `/narrator/*` endpoints and includes an explicit `/narrator/ask/legacy` route.
   - This indicates a migration/duplication state rather than a single canonical narrator surface.

2. **Root-level compatibility package and duplicate repo layout indicate transitional structure**.
   - Root `api/__init__.py` extends import path to `Algorithm/api` for compatibility.

3. **`/data/*` versioned-reality APIs appear orthogonal to the current prosperity snapshot path**.
   - They are functional but not used in primary smoke path or prosperity job orchestration.

4. **Static web UI is not main path yet**.
   - Presentational shell exists, but multiple panes are explicit placeholders.

## Immediate Fix Priorities

1. **Choose one official narrator API surface and deprecate the duplicate**.
   - Keep either `api/narrator/routes.py` or `api/llm/routes.py` as canonical, with compatibility aliases and deprecation timeline.

2. **Replace `correlation_guard_stub` with real implementation or remove from decision path until implemented**.
   - Current no-op can create false confidence in portfolio-risk controls.

3. **Remove/retire legacy stubs in prosperity jobs**.
   - `_release_job_lock` and `_insert_job_run` should be deleted or fully implemented behind clear migration flags.

4. **Harden DB-vs-live fallback boundaries**.
   - Strategy graph and signal flows should have explicit deterministic mode flags (DB-only vs provider-fallback) and observability when fallback is used.

5. **Publish one “official v1 flow” in docs and hide non-core paths in public docs**.
   - Promote only bootstrap → snapshot (or daily job) → latest signal/features (+ optional strategy graph).

