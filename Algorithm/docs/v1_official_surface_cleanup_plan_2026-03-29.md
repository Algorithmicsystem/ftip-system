# Official v1 Surface Cleanup Plan (2026-03-29)

## Official V1 Surface

The product should present exactly this path as the **official v1** sequence:

1. `POST /prosperity/bootstrap`
2. `POST /prosperity/snapshot/run`
3. `GET /prosperity/latest/signal`
4. `GET /prosperity/latest/features`
5. Optional scheduled equivalent: `POST /jobs/prosperity/daily-snapshot`

These endpoints already exist and are directly wired in the current API router stack.

---

## Non-Core / Legacy Areas

### A) Duplicate or parallel signal surfaces

- `/signals/latest`, `/signals/top`, `/signals/evidence` expose a second signal API namespace separate from the prosperity v1 reads.
- This duplicates user mental models for “latest signal” and creates namespace ambiguity (`/signals/latest` vs `/prosperity/latest/signal`).

**Disposition:** Document as non-v1 and de-emphasize in docs/UI now; defer removal until v2 compatibility window decision.

### B) Extra prosperity write/control endpoints outside v1 story

- `/prosperity/universe/upsert`
- `/prosperity/bars/ingest`, `/prosperity/bars/ingest_bulk`, `/prosperity/bars`
- `/prosperity/features/compute`, `/prosperity/signals/compute`
- `/prosperity/backtest`
- `/prosperity/coverage`, `/prosperity/graph/strategy`, `/prosperity/graph/universe`
- `/prosperity/health`

These are useful operational/developer tools but not required for the official v1 customer flow.

**Disposition:** Keep for internal/operator usage; mark non-v1 in docs and remove from “getting started” narratives.

### C) Non-v1 jobs surface sprawl

Beyond `POST /jobs/prosperity/daily-snapshot`, the jobs router contains cron aliases and status/coverage reads:

- `/jobs/prosperity/daily-snapshot/cron`
- `/jobs/prosperity/daily-snapshot/status`
- `/jobs/prosperity/daily-snapshot/summary`
- `/jobs/prosperity/daily-snapshot/coverage`
- `/jobs/prosperity/daily-snapshot/runs/{run_id}/coverage`

These are operational but broaden the exposed “story”.

**Disposition:** Keep hidden from v1 onboarding docs; classify as operator APIs.

### D) Entire feature families outside v1 scope

- Assistant/chat: `/assistant/*`
- Narrator/LLM: `/narrator/*`, `/narrator/signal`, `/narrator/portfolio`, `/narrator/ask/legacy`
- Backtest: `/backtest/*`
- Friction simulation: `/friction/simulate`
- Data warehouse ingestion/query: `/data/*`
- Market/features/signals batch jobs: `/jobs/data/*`, `/jobs/features/*`, `/jobs/signals/*`
- Provider diagnostics: `/providers/health`

**Disposition:** Explicitly document as non-v1 modules; defer productization/copy cleanup until after v1 launch.

---

## Confusing Product Surfaces

### 1) Root endpoint advertises many legacy endpoints

`GET /` returns a long static `endpoints` list that includes numerous non-v1 and legacy DB routes, which crowds out the official v1 story and can imply unsupported pathways.

### 2) Web app (`/app`) is not aligned to v1 story

The mounted UI is branded “FTIP Chat Orchestrator”, contains placeholder panels, and drives assistant/backtest endpoints (`/assistant/analyze`, `/assistant/narrate`, `/backtest/run`) rather than the official prosperity v1 path.

### 3) Naming collision around “latest signal”

Two different endpoint families communicate “latest signal” behavior:

- `GET /prosperity/latest/signal` (official v1)
- `GET /signals/latest` (legacy/parallel)

This is the most likely source of integration mistakes.

### 4) Optional job endpoint may look mandatory

Because jobs endpoints are numerous, users can infer scheduling APIs are primary rather than optional orchestration around `snapshot/run`.

---

## Recommended Cleanup Actions

## Tier 0 (minimum, safe, immediate)

1. **Publish an explicit v1 contract doc** in docs root with:
   - official sequence,
   - required headers/auth,
   - example request/response,
   - “non-v1 endpoints” appendix.
2. **Trim README “primary path” section** to only the five v1 endpoints.
3. **Add non-v1 labeling** to all other endpoint sections (“legacy/internal/non-v1”).
4. **Update root endpoint payload** so `endpoints` lists only v1 paths (+ `/health` and `/docs`), or split into `official_v1_endpoints` vs `other_endpoints`.
5. **Mark `/signals/*` as legacy** in OpenAPI descriptions/docstrings and cross-reference to `/prosperity/latest/signal`.

## Tier 1 (low risk, post-doc pass)

6. **Hide non-v1 routes from default docs view** (tagging strategy or docs curation).
7. **Rename/alias UI entrypoint copy** to “Internal demo UI (non-v1)” and remove from README quickstart.
8. **Consolidate jobs docs** so only `POST /jobs/prosperity/daily-snapshot` appears in v1 pages, with status/coverage routes in operator runbook.

## Tier 2 (deferred)

9. Decide whether `/signals/*` should be deprecated on a timeline (announce + removal date) or retained as permanent compatibility API.
10. Decide whether `/app` should be archived, rebuilt around the v1 flow, or moved to a separate internal tool surface.

---

## Minimum Safe V1 Presentation

If only the minimum viable cleanup is done, the product can still present a clear v1 story by doing **just these**:

- One canonical doc page titled “FTIP Official API v1” with only the five endpoints.
- README top-level quickstart that uses only those five endpoints.
- Root `/` endpoint response that explicitly highlights official v1 endpoints first.
- A short “Non-v1 APIs” section that lists assistant/narrator/backtest/data/jobs extras as internal or advanced.

This keeps implementation risk low (mostly docs and metadata changes), avoids breaking existing internal tooling, and removes ambiguity for new integrators.
