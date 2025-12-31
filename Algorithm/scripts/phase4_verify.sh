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

echo "== FTIP Phase 4 Verification =="
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
snapshot_body='{ "symbols":["AAPL","MSFT","NVDA","AMZN","TSLA"],"from_date":"2024-01-02","to_date":"2024-01-05","as_of_date":"2024-01-05","lookback":252,"concurrency":3,"compute_strategy_graph":true }'
snapshot_out="${tmpdir}/snapshot.json"
status=$(curl_json POST "/prosperity/snapshot/run" "$snapshot_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$snapshot_body")
if [[ "$status" != "200" ]]; then
  out_json "$snapshot_out"
  fail "/prosperity/snapshot/run HTTP ${status}"
fi
python - <<'PY' "$snapshot_out" || fail "/prosperity/snapshot/run invalid JSON"
import json, sys
body = json.load(open(sys.argv[1]))
rows = body.get("result", {}).get("rows_written", {})
if rows.get("signals") != 5 or rows.get("features") != 5:
    print(json.dumps(body, indent=2))
    sys.exit(1)
print("rows ok")
PY
pass "/prosperity/snapshot/run"

echo "-- Narrator strategy graph --"
strategy_body='{ "symbol":"AAPL","lookback":252,"days":365,"to_date":"2024-12-31" }'
strategy_out="${tmpdir}/narrator_strategy.json"
status=$(curl_json POST "/narrator/explain-strategy-graph" "$strategy_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$strategy_body")
if [[ "$status" != "200" ]]; then
  out_json "$strategy_out"
  fail "/narrator/explain-strategy-graph HTTP ${status}"
fi
python - <<'PY' "$strategy_out" || fail "/narrator/explain-strategy-graph invalid response"
import json, sys
body = json.load(open(sys.argv[1]))
if not body.get("explanation"):
    print(json.dumps(body, indent=2)); sys.exit(1)
if not body.get("graph", {}).get("nodes"):
    print(json.dumps(body, indent=2)); sys.exit(1)
print("strategy graph ok")
PY
pass "/narrator/explain-strategy-graph"

echo "-- Narrator diagnose --"
diagnose_out="${tmpdir}/diagnose.json"
status=$(curl_json POST "/narrator/diagnose" "$diagnose_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$diagnose_out"
  fail "/narrator/diagnose HTTP ${status}"
fi
python - <<'PY' "$diagnose_out" || fail "/narrator/diagnose invalid response"
import json, sys
body = json.load(open(sys.argv[1]))
checks = body.get("checks") or []
if not checks:
    print(json.dumps(body, indent=2)); sys.exit(1)
failed = [c for c in checks if c.get("status") != "pass"]
if failed:
    print(json.dumps(failed, indent=2)); sys.exit(1)
print("all checks pass")
PY
pass "/narrator/diagnose"

echo ""
echo "ALL CHECKS PASSED"
