#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:?BASE is required (e.g. https://ftip-system-production.up.railway.app)}"
KEY="${KEY:-}"
AUTH_HEADER=()
if [[ -n "${KEY}" ]]; then
  AUTH_HEADER=(-H "X-FTIP-API-Key: ${KEY}")
fi

tmpdir=$(mktemp -d)
out_json() { cat "$1"; }
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; exit 1; }

curl_json() {
  local method="$1"; shift
  local path="$1"; shift
  local outfile="$1"; shift
  local status
  status=$(curl -s -o "$outfile" -w "%{http_code}" -X "$method" "${AUTH_HEADER[@]}" "$@" "${BASE}${path}")
  echo "$status"
}

printf "== FTIP Phase 2 Verification ==\nBASE=%s\n\n" "$BASE"

version_file="${tmpdir}/version.json"
status=$(curl_json GET "/version" "$version_file")
if [[ "$status" != "200" ]]; then
  fail "/version returned HTTP ${status}"
fi
commit_sha=$(python - "$version_file" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("railway_git_commit_sha", "unknown"))
PY
)
pass "/version (commit: ${commit_sha})"

auth_file="${tmpdir}/auth_status.json"
auth_status_code=$(curl_json GET "/auth/status" "$auth_file")
if [[ "$auth_status_code" == "401" ]]; then
  if [[ -z "${KEY}" ]]; then
    fail "AUTH FAILED: Server requires KEY. Set KEY to a configured FTIP_API_KEY and retry."
  else
    fail "AUTH FAILED: Provided KEY not accepted by server. Check Railway FTIP_API_KEY and redeploy."
  fi
fi
if [[ "$auth_status_code" != "200" ]]; then
  fail "/auth/status returned HTTP ${auth_status_code}"
fi

AUTH_ENABLED=$(python - "$auth_file" <<'PY'
import json, sys
print(str(json.load(open(sys.argv[1])).get("auth_enabled", False)).lower())
PY
)
if [[ "$AUTH_ENABLED" == "true" && -z "${KEY}" ]]; then
  fail "AUTH FAILED: Server has auth enabled. Provide KEY=... environment variable."
fi

prosperity_health="${tmpdir}/prosperity_health.json"
status=$(curl_json GET "/prosperity/health" "$prosperity_health")
[[ "$status" == "200" ]] || fail "/prosperity/health HTTP ${status}"
pass "/prosperity/health"

snapshot_file="${tmpdir}/snapshot.json"
snapshot_payload='{
  "symbols":["AAPL","MSFT","NVDA","AMZN","TSLA"],
  "from_date":"2023-01-01",
  "to_date":"2024-12-31",
  "as_of_date":"2024-12-31",
  "lookback":252,
  "concurrency":3
}'
status=$(curl_json POST "/prosperity/snapshot/run" "$snapshot_file" -H "Content-Type: application/json" -d "$snapshot_payload")
if [[ "$status" != "200" ]]; then
  out_json "$snapshot_file"
  fail "/prosperity/snapshot/run HTTP ${status}"
fi

python - <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
rows = body.get("result", {}).get("rows_written", {})
if not (rows.get("signals", 0) > 0 and rows.get("features", 0) > 0):
    print(json.dumps(body, indent=2))
    sys.exit(1)
PY
"$snapshot_file" || fail "/prosperity/snapshot/run missing rows_written"
pass "/prosperity/snapshot/run"

latest_signal_file="${tmpdir}/latest_signal.json"
status=$(curl_json GET "/prosperity/latest/signal?symbol=AAPL&lookback=252" "$latest_signal_file")
if [[ "$status" != "200" ]]; then
  out_json "$latest_signal_file"
  fail "/prosperity/latest/signal HTTP ${status}"
fi
pass "/prosperity/latest/signal"

graph_strategy_file="${tmpdir}/graph_strategy.json"
status=$(curl_json GET "/prosperity/graph/strategy?symbol=AAPL&lookback=252&days=365" "$graph_strategy_file")
if [[ "$status" != "200" ]]; then
  out_json "$graph_strategy_file"
  fail "/prosperity/graph/strategy HTTP ${status}"
fi
pass "/prosperity/graph/strategy"

graph_universe_file="${tmpdir}/graph_universe.json"
status=$(curl_json GET "/prosperity/graph/universe" "$graph_universe_file")
if [[ "$status" != "200" ]]; then
  out_json "$graph_universe_file"
  fail "/prosperity/graph/universe HTTP ${status}"
fi
pass "/prosperity/graph/universe"

rm -rf "$tmpdir"

