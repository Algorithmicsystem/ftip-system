#!/usr/bin/env bash
set -euo pipefail

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

echo "== FTIP Phase 6 Verification =="
echo "BASE=${BASE}";
echo ""

echo "-- Public endpoints --"
health_file="${tmpdir}/health.json"
status=$(curl_json GET "/health" "$health_file")
if [[ "$status" != "200" ]]; then
  out_json "$health_file"
  fail "/health HTTP ${status}"
fi
pass "/health"

version_file="${tmpdir}/version.json"
status=$(curl_json GET "/version" "$version_file")
if [[ "$status" != "200" ]]; then
  out_json "$version_file"
  fail "/version HTTP ${status}"
fi
pass "/version"

echo "-- Auth guards --"
unauth_narrator=$(curl -s -i "${BASE}/narrator/health" | head -n 1)
if ! printf '%s' "$unauth_narrator" | grep -q "401"; then
  echo "$unauth_narrator"
  fail "/narrator/health should require API key"
fi
pass "/narrator/health without key"

auth_narrator="${tmpdir}/narrator_health.json"
status=$(curl_json GET "/narrator/health" "$auth_narrator" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$auth_narrator"
  fail "/narrator/health HTTP ${status} with key"
fi
pass "/narrator/health with key"

unauth_prosperity=$(curl -s -i "${BASE}/prosperity/health" | head -n 1)
if ! printf '%s' "$unauth_prosperity" | grep -q "401"; then
  echo "$unauth_prosperity"
  fail "/prosperity/health should require API key"
fi
pass "/prosperity/health without key"

auth_prosperity="${tmpdir}/prosperity_health.json"
status=$(curl_json GET "/prosperity/health" "$auth_prosperity" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$auth_prosperity"
  fail "/prosperity/health HTTP ${status} with key"
fi
pass "/prosperity/health with key"

echo "-- Daily snapshot job --"
job_out="${tmpdir}/daily_job.json"
status=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$job_out"
  fail "/jobs/prosperity/daily-snapshot HTTP ${status}"
fi
if ! jq -e '(.rows_written.signals // 0) > 0' "$job_out" >/dev/null; then
  out_json "$job_out"
  fail "daily snapshot expected signals rows_written > 0"
fi
pass "/jobs/prosperity/daily-snapshot"

echo "-- Latest signal --"
latest_signal="${tmpdir}/latest_signal.json"
status=$(curl_json GET "/prosperity/latest/signal?symbol=AAPL&lookback=252" "$latest_signal" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$latest_signal"
  fail "/prosperity/latest/signal HTTP ${status}"
fi
pass "/prosperity/latest/signal"

echo "-- Strategy graph sample --"
strategy_graph="${tmpdir}/strategy_graph.json"
status=$(curl_json GET "/prosperity/graph/strategy?symbol=AAPL&lookback=252&days=365" "$strategy_graph" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$strategy_graph"
  fail "/prosperity/graph/strategy HTTP ${status}"
fi
if ! jq -e '(.series_sample // []) | length > 0' "$strategy_graph" >/dev/null; then
  out_json "$strategy_graph"
  fail "/prosperity/graph/strategy expected non-empty series_sample"
fi
pass "/prosperity/graph/strategy"

echo ""
echo "ALL CHECKS PASSED"
