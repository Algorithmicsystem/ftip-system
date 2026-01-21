#!/usr/bin/env sh
set -eu

if [ -z "${DATABASE_URL:-}" ]; then
  echo "DATABASE_URL is required for smoke run"
  exit 1
fi

export FTIP_DB_ENABLED=1
export FTIP_DB_READ_ENABLED=1
export FTIP_DB_WRITE_ENABLED=1

python3 - <<'PY'
from api import migrations

migrations.ensure_schema()
print("migrations ok")
PY

uvicorn api.main:app --host 0.0.0.0 --port 8000 >/tmp/ftip_smoke.log 2>&1 &
PID=$!
sleep 2

curl -s -X POST http://localhost:8000/assistant/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","horizon":"swing","risk_mode":"balanced"}' >/tmp/ftip_analyze.json || true

curl -s -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","universe":"custom","date_start":"2024-01-01","date_end":"2024-01-10","horizon":"swing","risk_mode":"balanced","signal_version_hash":"auto","cost_model":{"fee_bps":1,"slippage_bps":5}}' \
  >/tmp/ftip_backtest.json || true

kill $PID
wait $PID 2>/dev/null || true

echo "Smoke outputs:"
cat /tmp/ftip_analyze.json
cat /tmp/ftip_backtest.json
