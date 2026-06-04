# AXIOM Intelligence Platform — Deployment Guide

## Section 1: Local Development Setup

### Prerequisites
- Python 3.9+
- PostgreSQL 16 (optional — system works without DB in read-only mode)

### Steps

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template
cp .env.example .env        # then edit .env with your keys

# 4. Run the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The dashboard is served at `http://localhost:8000/app`.

---

## Section 2: Docker Compose Deployment

```bash
# From the repo root (ftip-system/)
cp .env.example .env        # configure secrets

docker compose up --build -d

# Check logs
docker compose logs -f api

# Stop
docker compose down
```

The API will be available at `http://localhost:8000`.
PostgreSQL data is persisted in the `postgres_data` Docker volume.

---

## Section 3: Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes (production) | PostgreSQL connection string. Format: `postgresql://user:pass@host:5432/db` |
| `FTIP_API_KEY` | Yes (production) | Master API key for internal authentication |
| `POLYGON_API_KEY` | Yes (production) | Polygon.io market data API key |
| `ANTHROPIC_API_KEY` | Recommended | Enables LLM-powered narrative generation |
| `FRED_API_KEY` | Recommended | Federal Reserve Economic Data |
| `FTIP_DB_ENABLED` | Production | `1` to enable PostgreSQL persistence |
| `FTIP_DB_WRITE_ENABLED` | Production | `1` to allow write operations |
| `FTIP_DB_READ_ENABLED` | Production | `1` to allow read operations (default: `1`) |
| `FTIP_DB_REQUIRED` | Production | `1` to fail fast if DB unavailable at startup |
| `FTIP_MIGRATIONS_AUTO` | Production | `1` to run pending migrations at startup |
| `FTIP_SCHEDULER_ENABLED` | Production | `1` to enable background job scheduler |
| `FTIP_LLM_ENABLED` | Optional | `1` to enable LLM narrative features |
| `AXIOM_ENV` | Optional | `development` \| `staging` \| `production` (default: `development`) |
| `PAGERDUTY_KEY` | Optional | PagerDuty integration key for critical alerts |
| `SLACK_WEBHOOK` | Optional | Slack webhook URL for monitoring notifications |
| `POSTGRES_DB` | Docker only | PostgreSQL database name (default: `axiom`) |
| `POSTGRES_USER` | Docker only | PostgreSQL user (default: `axiom`) |
| `POSTGRES_PASSWORD` | Docker only | PostgreSQL password |

---

## Section 4: Running Migrations

Migrations run automatically when `FTIP_MIGRATIONS_AUTO=1`. To run manually:

```bash
# Via Python
python -c "from api.migrations.runner import run_migrations; run_migrations()"

# Or trigger via API
curl -X POST http://localhost:8000/db/migrate
```

Migrations are idempotent — safe to re-run. Current migration count: 95+.

To check migration status:
```bash
curl http://localhost:8000/db/health | jq .
```

---

## Section 5: Scheduler Configuration

The scheduler runs 9 background jobs when `FTIP_SCHEDULER_ENABLED=1`:

| Job | Schedule | Description |
|---|---|---|
| `morning_briefing` | 7:00 AM weekdays | Generate morning intelligence briefing |
| `intraday_update_10` | 10:30 AM weekdays | Intraday signal update |
| `intraday_ic_10` | 10:35 AM weekdays | IC computation update |
| `intraday_update_12` | 12:30 PM weekdays | Midday signal update |
| `intraday_update_14` | 2:30 PM weekdays | Afternoon signal update |
| `intraday_update_16` | 4:30 PM weekdays | Close signal update |
| `full_daily_pipeline` | 5:00 PM weekdays | Full AXIOM pipeline run |
| `ml_training_check` | 6:00 PM weekdays | ML model drift check / retraining |
| `memory_consolidation` | 7:00 PM weekdays | Memory & cache consolidation |

Scheduler status: `GET /jobs/scheduler/status`

---

## Section 6: Performance Tuning Guide

### Database Pool
Adjust via `AXIOM_ENV` (see Section 3):
- `development`: pool_size=5, max=10
- `staging`: pool_size=10, max=20
- `production`: pool_size=20, max=50

Monitor pool: `GET /cloud/db/pool-stats`

### Worker Count
Set `--workers` in the uvicorn CMD. Recommendation: `2 × CPU_cores`.
The Dockerfile defaults to 4 workers.

### Cache TTL
Intelligence cache TTL is 30 minutes in production. Warm with:
```bash
curl http://localhost:8000/intelligence/universal/AAPL
```

### Performance Report
```bash
curl http://localhost:8000/cloud/performance/report | jq .
```

---

## Section 7: Monitoring Setup

### Production Health
```bash
curl http://localhost:8000/cloud/monitoring/health | jq .
```

### Production Readiness Check
```bash
curl http://localhost:8000/cloud/readiness | jq .
curl http://localhost:8000/cloud/readiness/summary | jq .
```

### Alert Thresholds
| Metric | Warning | Critical |
|---|---|---|
| API error rate | 1% | 5% |
| API p99 latency | 500ms | 2000ms |
| DB pool utilisation | 70% | 90% |
| Pipeline failures (24h) | 1 | 3 |
| ML PSI score | 0.15 | 0.25 |
| SRI risk level | 60 | 80 |
| Data staleness | 25h | 48h |

### PagerDuty Integration
Set `PAGERDUTY_KEY` in environment. Alerts with `severity="page"` will trigger an incident.

### Slack Integration
Set `SLACK_WEBHOOK` in environment. Alerts with `severity="critical"` will post to Slack.

### System Health Endpoint
`GET /orchestration/health` — returns overall health score, IC state, and component breakdown.
