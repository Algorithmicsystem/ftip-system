#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:?BASE is required (e.g. https://ftip-system-production.up.railway.app)}"
KEY="${KEY:?KEY is required (X-FTIP-API-Key) for narrator endpoints}"
AUTH_HEADER=(-H "X-FTIP-API-Key: ${KEY}")

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

tmpdir=$(mktemp -d)
printf "== FTIP Phase 3 Verification ==\nBASE=%s\n\n" "$BASE"

health_file="${tmpdir}/health.json"
status=$(curl_json GET "/health" "$health_file")
[[ "$status" == "200" ]] || fail "/health HTTP ${status}"
pass "/health"

version_file="${tmpdir}/version.json"
status=$(curl_json GET "/version" "$version_file")
[[ "$status" == "200" ]] || fail "/version HTTP ${status}"
pass "/version"

narrator_health_no_auth="${tmpdir}/narrator_health_no_auth.json"
status=$(curl_json GET "/narrator/health" "$narrator_health_no_auth")
if [[ "$status" != "401" ]]; then
  out_json "$narrator_health_no_auth"
  fail "/narrator/health expected 401 without key (got ${status})"
fi
pass "/narrator/health without key returns 401"

narrator_health="${tmpdir}/narrator_health.json"
status=$(curl_json GET "/narrator/health" "$narrator_health" "${AUTH_HEADER[@]}")
[[ "$status" == "200" ]] || { out_json "$narrator_health"; fail "/narrator/health HTTP ${status}"; }
HAS_KEY=$(python - "$narrator_health" <<'PY'
import json, sys
print(str(bool(json.load(open(sys.argv[1])).get("has_api_key"))).lower())
PY
)
pass "/narrator/health with key (has_api_key=${HAS_KEY})"

ask_body='{ "question":"Why is AAPL interesting?","symbols":["AAPL","MSFT"],"as_of_date":"2024-12-31","lookback":252,"days":365 }'
ask_out="${tmpdir}/narrator_ask.json"
status=$(curl_json POST "/narrator/ask" "$ask_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$ask_body")
if [[ "$status" != "200" ]]; then
  out_json "$ask_out"
  fail "/narrator/ask HTTP ${status}"
fi
python - <<'PY'
import json, sys
body=json.load(open(sys.argv[1]))
answer=body.get("answer")
if not answer:
    print(json.dumps(body, indent=2))
    sys.exit(1)
PY "$ask_out" || fail "/narrator/ask missing answer"
pass "/narrator/ask"

explain_body='{ "symbol":"AAPL","as_of_date":"2024-12-31","lookback":252,"days":365 }'
explain_out="${tmpdir}/narrator_explain.json"
status=$(curl_json POST "/narrator/explain-signal" "$explain_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$explain_body")
if [[ "$status" != "200" ]]; then
  out_json "$explain_out"
  fail "/narrator/explain-signal HTTP ${status}"
fi
python - <<'PY'
import json, sys
body=json.load(open(sys.argv[1]))
if not (body.get("explanation") and body.get("symbol")):
    print(json.dumps(body, indent=2))
    sys.exit(1)
PY "$explain_out" || fail "/narrator/explain-signal missing explanation"
pass "/narrator/explain-signal"

rm -rf "$tmpdir"
