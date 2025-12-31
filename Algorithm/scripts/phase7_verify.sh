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

echo "== FTIP Phase 7 Verification =="
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
job_no_key=$(curl -s -i "${BASE}/jobs/prosperity/daily-snapshot" | head -n 1)
if ! printf '%s' "$job_no_key" | grep -q "401"; then
  echo "$job_no_key"
  fail "/jobs/prosperity/daily-snapshot should require API key"
fi
pass "/jobs/prosperity/daily-snapshot without key"

echo "-- Daily snapshot job lock + status --"
job_out="${tmpdir}/daily_job.json"
status=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$job_out"
  fail "first /jobs/prosperity/daily-snapshot HTTP ${status}"
fi
if ! jq -e '.run_id and (.rows_written.signals // 0) >= 0' "$job_out" >/dev/null; then
  out_json "$job_out"
  fail "daily snapshot response missing run_id or rows_written"
fi
pass "first /jobs/prosperity/daily-snapshot"

job_again="${tmpdir}/daily_job_again.json"
status=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_again" "${AUTH_HEADER[@]}")
if [[ "$status" != "409" ]]; then
  out_json "$job_again"
  fail "second /jobs/prosperity/daily-snapshot expected 409 got ${status}"
fi
pass "second /jobs/prosperity/daily-snapshot lock"

status_out="${tmpdir}/job_status.json"
status=$(curl_json GET "/jobs/prosperity/daily-snapshot/status" "$status_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$status_out"
  fail "/jobs/prosperity/daily-snapshot/status HTTP ${status}"
fi
if ! jq -e '.status and .started_at and .run_id' "$status_out" >/dev/null; then
  out_json "$status_out"
  fail "status endpoint missing fields"
fi
pass "/jobs/prosperity/daily-snapshot/status"

echo ""
echo "ALL CHECKS PASSED"
