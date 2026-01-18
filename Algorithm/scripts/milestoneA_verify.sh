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
  local max_time="$1"; shift
  local HTTP_CODE
  HTTP_CODE=$(curl -s -o "$outfile" -w "%{http_code}" --max-time "$max_time" -X "$method" "$@" "${BASE}${path}")
  echo "$HTTP_CODE"
}

parse_field() {
  local file="$1"; shift
  local field="$1"; shift
  python3 - <<'PY' "$file" "$field"
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

validate_failure_shape() {
  local file="$1"
  python3 - <<'PY' "$file"
import json, sys
payload = json.load(open(sys.argv[1]))
failed = payload.get("symbols_failed") or []
for item in failed:
    if not item.get("reason_code"):
        sys.exit("missing reason_code")
    if item.get("attempts") is None:
        sys.exit("missing attempts")
    if item.get("retryable") is None:
        sys.exit("missing retryable")
print("ok")
PY
}

ensure_failed_symbols_reason_code() {
  local file="$1"
  python3 - <<'PY' "$file"
import json, sys
payload = json.load(open(sys.argv[1]))
failed = payload.get("failed_symbols") or []
for item in failed:
    if not item.get("reason_code"):
        sys.exit("missing reason_code")
print("ok")
PY
}

echo "== FTIP Milestone A Verification =="
echo "BASE=${BASE}";
echo ""

echo "-- Public endpoints --"
health_file="${tmpdir}/health.json"
HTTP_CODE=$(curl_json GET "/health" "$health_file" 30)
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$health_file"
  fail "/health HTTP ${HTTP_CODE}"
fi
pass "/health"

version_file="${tmpdir}/version.json"
HTTP_CODE=$(curl_json GET "/version" "$version_file" 30)
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$version_file"
  fail "/version HTTP ${HTTP_CODE}"
fi
pass "/version"

echo "-- Auth guards --"
job_no_key_out="${tmpdir}/job_no_key.json"
HTTP_CODE=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_no_key_out" 30)
if [[ "$HTTP_CODE" != "401" ]]; then
  out_json "$job_no_key_out"
  fail "/jobs/prosperity/daily-snapshot without key expected 401 got ${HTTP_CODE}"
fi
pass "/jobs/prosperity/daily-snapshot without key"

job_first_out="${tmpdir}/daily_job.json"
HTTP_CODE=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_first_out" 240 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$job_first_out"
  fail "/jobs/prosperity/daily-snapshot with key HTTP ${HTTP_CODE}"
fi

if ! validate_failure_shape "$job_first_out" >/dev/null; then
  out_json "$job_first_out"
  fail "daily snapshot response missing failure fields"
fi

mapfile -t parsed_run < <(python3 - <<'PY' "$job_first_out"
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
HTTP_CODE=$(curl_json POST "/jobs/prosperity/daily-snapshot" "$job_again" 240 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "409" ]]; then
  out_json "$job_again"
  fail "second /jobs/prosperity/daily-snapshot expected 409 got ${HTTP_CODE}"
fi
pass "second /jobs/prosperity/daily-snapshot lock"

status_out="${tmpdir}/job_status.json"
HTTP_CODE=$(curl_json GET "/jobs/prosperity/daily-snapshot/status" "$status_out" 30 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$status_out"
  fail "/jobs/prosperity/daily-snapshot/status HTTP ${HTTP_CODE}"
fi
if [[ -z "$(parse_field "$status_out" run_id || true)" ]]; then
  out_json "$status_out"
  fail "status endpoint missing run_id"
fi
pass "/jobs/prosperity/daily-snapshot/status"

echo "-- Coverage endpoints --"
coverage_out="${tmpdir}/coverage.json"
HTTP_CODE=$(curl_json GET "/jobs/prosperity/daily-snapshot/coverage?as_of_date=${as_of}" "$coverage_out" 30 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$coverage_out"
  fail "coverage by date HTTP ${HTTP_CODE}"
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
HTTP_CODE=$(curl_json GET "/jobs/prosperity/daily-snapshot/runs/${run_id}/coverage" "$run_cov_out" 30 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$run_cov_out"
  fail "coverage by run HTTP ${HTTP_CODE}"
fi
if [[ "$(parse_field "$run_cov_out" run_id)" != "$run_id" ]]; then
  out_json "$run_cov_out"
  fail "coverage run_id mismatch"
fi
pass "/jobs/prosperity/daily-snapshot/runs/{run_id}/coverage"

summary_out="${tmpdir}/summary.json"
HTTP_CODE=$(curl_json GET "/jobs/prosperity/daily-snapshot/summary" "$summary_out" 30 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "200" ]]; then
  out_json "$summary_out"
  fail "/jobs/prosperity/daily-snapshot/summary HTTP ${HTTP_CODE}"
fi
pass "/jobs/prosperity/daily-snapshot/summary"

cron_out="${tmpdir}/cron.json"
HTTP_CODE=$(curl_json POST "/jobs/prosperity/daily-snapshot/cron" "$cron_out" 30 "${AUTH_HEADER[@]}")
if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "409" ]]; then
  out_json "$cron_out"
  fail "/jobs/prosperity/daily-snapshot/cron expected 200/409 got ${HTTP_CODE}"
fi
pass "/jobs/prosperity/daily-snapshot/cron"

echo ""
echo "ALL MILESTONE A CHECKS PASSED"
