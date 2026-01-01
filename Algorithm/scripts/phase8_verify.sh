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

parse_field() {
  local file="$1"; shift
  local field="$1"; shift
  python - <<'PY' "$file" "$field"
import json, sys
file_path, field = sys.argv[1:3]
data = json.load(open(file_path))
parts = field.split('.')
val = data
for part in parts:
    if isinstance(val, dict) and part in val:
        val = val[part]
    else:
        sys.exit("")
if isinstance(val, (dict, list)):
    sys.exit(json.dumps(val))
print(val)
PY
}

validate_failure_reasons() {
  local file="$1"
  python - <<'PY' "$file"
import json, sys
payload = json.load(open(sys.argv[1]))
failed = payload.get("symbols_failed") or []
for item in failed:
    if not item.get("reason_code"):
        sys.exit("missing reason_code")
print("ok")
PY
}

ensure_failed_symbols_reason_code() {
  local file="$1"
  python - <<'PY' "$file"
import json, sys
payload = json.load(open(sys.argv[1]))
failed = payload.get("failed_symbols") or []
for item in failed:
    if not item.get("reason_code"):
        sys.exit("missing reason_code")
print("ok")
PY
}

echo "== FTIP Phase 8 Verification =="
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
job_no_key_out="${tmpdir}/job_no_key.json"
status=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_no_key_out")
if [[ "$status" != "401" ]]; then
  out_json "$job_no_key_out"
  fail "/jobs/prosperity/daily-snapshot without key expected 401 got ${status}"
fi
pass "/jobs/prosperity/daily-snapshot without key"

job_first_out="${tmpdir}/daily_job.json"
status=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_first_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$job_first_out"
  fail "/jobs/prosperity/daily-snapshot with key HTTP ${status}"
fi

if ! validate_failure_reasons "$job_first_out" >/dev/null; then
  out_json "$job_first_out"
  fail "daily snapshot response missing reason codes"
fi

mapfile -t parsed_run < <(python - <<'PY' "$job_first_out"
import json, sys
payload = json.load(open(sys.argv[1]))
run_id = payload.get("run_id")
as_of = payload.get("as_of_date")
rows_written = payload.get("rows_written")
if not run_id or rows_written is None:
    sys.exit("missing run_id or rows_written")
print(run_id)
print(as_of or "")
PY
)
if [[ ${#parsed_run[@]} -lt 2 ]]; then
  out_json "$job_first_out"
  fail "unable to parse run_id/as_of_date"
fi
run_id=${parsed_run[0]}
as_of=${parsed_run[1]}
pass "/jobs/prosperity/daily-snapshot with key"

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
if [[ -z "$(parse_field "$status_out" run_id || true)" ]]; then
  out_json "$status_out"
  fail "status endpoint missing run_id"
fi
pass "/jobs/prosperity/daily-snapshot/status"

echo "-- Coverage endpoints --"
coverage_out="${tmpdir}/coverage.json"
status=$(curl_json GET "/jobs/prosperity/daily-snapshot/coverage?as_of_date=${as_of}" "$coverage_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$coverage_out"
  fail "coverage by date HTTP ${status}"
fi
if [[ -z "$(parse_field "$coverage_out" attempted || true)" ]]; then
  out_json "$coverage_out"
  fail "coverage payload missing attempted"
fi
if ! ensure_failed_symbols_reason_code "$coverage_out" >/dev/null; then
  out_json "$coverage_out"
  fail "coverage failed_symbols missing reason_code"
fi
pass "/jobs/prosperity/daily-snapshot/coverage"

run_cov_out="${tmpdir}/run_coverage.json"
status=$(curl_json GET "/jobs/prosperity/daily-snapshot/runs/${run_id}/coverage" "$run_cov_out" "${AUTH_HEADER[@]}")
if [[ "$status" != "200" ]]; then
  out_json "$run_cov_out"
  fail "coverage by run HTTP ${status}"
fi
if [[ "$(parse_field "$run_cov_out" run_id)" != "$run_id" ]]; then
  out_json "$run_cov_out"
  fail "coverage run_id mismatch"
fi
pass "/jobs/prosperity/daily-snapshot/runs/{run_id}/coverage"

echo ""
echo "ALL CHECKS PASSED"
