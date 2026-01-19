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

echo "== FTIP Phase 3 Verification =="
echo "BASE=${BASE}"
echo ""

echo "-- Public endpoints --"
health_file="${tmpdir}/health.json"
status=$(curl_json GET "/health" "$health_file")
if [[ "$status" != "200" ]]; then
  out_json "$health_file"
  fail "/health HTTP ${status}"
fi
HEALTH_STATUS=$(python - "$health_file" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status"))
PY
)
[[ "$HEALTH_STATUS" == "ok" ]] || fail "/health missing status ok"
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

echo "-- Narrator ask --"
ask_body='{ "question":"Why is AAPL interesting?","symbols":["AAPL","MSFT"],"as_of_date":"2024-12-31","lookback":252,"days":365 }'
ask_out="${tmpdir}/narrator_ask.json"
status=$(curl_json POST "/narrator/ask" "$ask_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$ask_body")
if [[ "$status" != "200" ]]; then
  out_json "$ask_out"
  fail "/narrator/ask HTTP ${status}"
fi
python3 - <<'PY' "$ask_out" || fail "/narrator/ask missing answer"
import json, sys
body = json.load(open(sys.argv[1]))
if not body.get("answer"):
    print(json.dumps(body, indent=2))
    sys.exit(1)
print("answer present")
PY
echo "Response preview:"; head -c 400 "$ask_out"; echo ""
pass "/narrator/ask"

echo "-- Narrator explain-signal --"
explain_body='{"symbol":"AAPL","as_of_date":"2024-12-31","signal":{"action":"BUY","confidence":0.6,"reason_codes":["TREND_UP"],"stop_loss":90},"features":{},"quality":{"sentiment_ok":true,"intraday_ok":false,"fundamentals_ok":true},"bars":{},"sentiment":{"headline_count":2}}'
explain_out="${tmpdir}/narrator_explain.json"
status=$(curl_json POST "/narrator/explain-signal" "$explain_out" "${AUTH_HEADER[@]}" -H "Content-Type: application/json" -d "$explain_body")
if [[ "$status" != "200" ]]; then
  out_json "$explain_out"
  fail "/narrator/explain-signal HTTP ${status}"
fi
python3 - <<'PY' "$explain_out" || fail "/narrator/explain-signal invalid response"
import json, sys
body = json.load(open(sys.argv[1]))
if not body.get("explanation"):
    print(json.dumps(body, indent=2))
    sys.exit(1)
print("explanation present")
PY
echo "Response preview:"; head -c 400 "$explain_out"; echo ""
pass "/narrator/explain-signal"
