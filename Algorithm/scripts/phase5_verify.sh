#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:?BASE is required (e.g. https://ftip-system-production.up.railway.app)}"
KEY="${KEY:?KEY is required (X-FTIP-API-Key) for narrator endpoints}"
AUTH_HEADER=(-H "X-FTIP-API-Key: ${KEY}")

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

out_json() { cat "$1"; }
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; exit 1; }

curl_json() {
  local method="$1"; shift
  local path="$1"; shift
  local outfile="$1"; shift
  local status
  status=$(curl -s -o "$outfile" -w "%{http_code}" -X "$method" "$@" "${BASE}${path}")
  echo "$status"
}

echo "== FTIP Phase 5 Verification =="
echo "BASE=${BASE}"
echo ""

echo "-- Public endpoints --"
health_file="${tmpdir}/health.json"
status=$(curl_json GET "/health" "$health_file")
if [[ "$status" != "200" ]]; then
  out_json "$health_file"
  fail "/health HTTP ${status}"
fi
if ! grep -q '"status"' "$health_file"; then
  out_json "$health_file"
  fail "/health missing status"
fi
pass "/health"

version_file="${tmpdir}/version.json"
status=$(curl_json GET "/version" "$version_file")
if [[ "$status" != "200" ]]; then
  out_json "$version_file"
  fail "/version HTTP ${status}"
fi
python - <<'PY' "$version_file" || fail "/version did not return JSON"
import json, sys
json.load(open(sys.argv[1]))
print("version JSON ok")
PY
pass "/version"


echo "-- Narrator auth --"
echo "Expect 401 without key:"
unauth_resp=$(curl -s -i "${BASE}/narrator/health" | head -n 20)
status_line=$(printf '%s\n' "$unauth_resp" | head -n 1)
if ! printf '%s' "$status_line" | grep -q "401"; then
  printf '%s\n' "$unauth_resp"
  fail "/narrator/health should return 401 without key"
fi
echo "${status_line}"
pass "/narrator/health without key"

narrator_health="${tmpdir}/narrator_health.json"
status=$(curl_json GET "/narrator/health" "$narrator_health" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$narrator_health"
  fail "/narrator/health HTTP ${status}"
fi
HAS_STATUS=$(python - "$narrator_health" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
print(body.get("status"), body.get("has_api_key"))
PY
)
if [[ "$HAS_STATUS" != "ok True" ]]; then
  out_json "$narrator_health"
  fail "/narrator/health missing status ok or has_api_key true"
fi
pass "/narrator/health with key"


echo "-- Snapshot run --"
snapshot_body='{ "symbols":["AAPL","MSFT","NVDA","AMZN","TSLA"],"from_date":"2020-01-01","to_date":"2024-12-31","as_of_date":"2024-12-31","lookback":252,"concurrency":3,"compute_strategy_graph":true }'
snapshot_out="${tmpdir}/snapshot.json"
status=$(curl_json POST "/prosperity/snapshot/run" "$snapshot_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$snapshot_body")
if [[ "$status" != "200" ]]; then
  out_json "$snapshot_out"
  fail "/prosperity/snapshot/run HTTP ${status}"
fi
if ! jq . "$snapshot_out" >/dev/null 2>&1; then
  out_json "$snapshot_out"
  fail "/prosperity/snapshot/run invalid JSON"
fi
if jq -e '((.result.symbols_failed // []) | map([.reasons[]? // ""][] | ascii_downcase | contains("insufficient bars")) | any)' "$snapshot_out" >/dev/null; then
  out_json "$snapshot_out"
  fail "/prosperity/snapshot/run insufficient bars for lookback=252; extend date range to include enough history"
fi
if ! jq -e '(.result.rows_written.signals // 0) > 0' "$snapshot_out" >/dev/null; then
  out_json "$snapshot_out"
  fail "/prosperity/snapshot/run rows_written.signals expected > 0"
fi
pass "/prosperity/snapshot/run"


echo "-- Narrator portfolio (include_backtest=false) --"
portfolio_payload_no_bt='{ "symbols":["AAPL","MSFT"],"from_date":"2024-01-01","to_date":"2024-06-30","lookback":252,"rebalance_every":5,"include_backtest":false }'
portfolio_no_bt="${tmpdir}/portfolio_no_bt.json"
status=$(curl_json POST "/narrator/portfolio" "$portfolio_no_bt" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$portfolio_payload_no_bt")
if [[ "$status" != "200" ]]; then
  out_json "$portfolio_no_bt"
  fail "/narrator/portfolio (no backtest) HTTP ${status}"
fi
python - <<'PY' "$portfolio_no_bt" || fail "/narrator/portfolio (no backtest) invalid JSON"
import json, sys
body = json.load(open(sys.argv[1]))
perf = body.get("performance") or {}
for key in ("return", "sharpe", "max_drawdown", "turnover"):
    if key not in perf:
        raise SystemExit(f"missing performance key: {key}")
    val = perf[key]
    if val is None or not isinstance(val, (int, float)):
        raise SystemExit(f"performance {key} not numeric: {val}")
print("portfolio without backtest ok")
PY
pass "/narrator/portfolio include_backtest=false"


echo "-- Narrator portfolio (include_backtest=true) --"
portfolio_payload_bt='{ "symbols":["AAPL","MSFT"],"from_date":"2023-01-01","to_date":"2024-12-31","lookback":252,"rebalance_every":5,"include_backtest":true }'
portfolio_bt="${tmpdir}/portfolio_bt.json"
status=$(curl_json POST "/narrator/portfolio" "$portfolio_bt" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$portfolio_payload_bt")
if [[ "$status" != "200" ]]; then
  out_json "$portfolio_bt"
  fail "/narrator/portfolio (backtest) HTTP ${status}"
fi
python - <<'PY' "$portfolio_bt" || fail "/narrator/portfolio (backtest) invalid JSON"
import json, sys
body = json.load(open(sys.argv[1]))
perf = body.get("performance") or {}
for key in ("return", "sharpe", "max_drawdown", "turnover"):
    if key not in perf:
        raise SystemExit(f"missing performance key: {key}")
    val = perf[key]
    if val is None or not isinstance(val, (int, float)):
        raise SystemExit(f"performance {key} not numeric: {val}")
print("portfolio with backtest ok")
PY
pass "/narrator/portfolio include_backtest=true"

echo ""
echo "ALL CHECKS PASSED"
