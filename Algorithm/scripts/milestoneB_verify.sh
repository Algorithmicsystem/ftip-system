#!/usr/bin/env sh
set -eu

if [ -z "${BASE:-}" ] || [ -z "${KEY:-}" ]; then
  echo "BASE and KEY env vars required" >&2
  exit 1
fi

AUTH_HEADER="X-FTIP-API-Key: ${KEY}"

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

out_json() { cat "$1"; }
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; exit 1; }

curl_json() {
  METHOD="$1"
  PATH="$2"
  OUTFILE="$3"
  MAX_TIME="$4"
  HEADER="$5"
  if [ -n "$HEADER" ]; then
    HTTP_CODE=$(curl -s -o "$OUTFILE" -w "%{http_code}" --max-time "$MAX_TIME" -X "$METHOD" -H "$HEADER" "$BASE$PATH")
  else
    HTTP_CODE=$(curl -s -o "$OUTFILE" -w "%{http_code}" --max-time "$MAX_TIME" -X "$METHOD" "$BASE$PATH")
  fi
  echo "$HTTP_CODE"
}

parse_field() {
  FILE="$1"
  FIELD="$2"
  python3 - <<'PY' "$FILE" "$FIELD"
import json
import sys
file_path, field = sys.argv[1:3]
value = json.load(open(file_path))
parts = field.split('.')
for part in parts:
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        sys.exit("")
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

ensure_failed_reason_codes() {
  FILE="$1"
  python3 - <<'PY' "$FILE"
import json
import sys
payload = json.load(open(sys.argv[1]))
failed = payload.get("symbols_failed") or []
for item in failed:
    if not item.get("reason_code"):
        sys.exit("missing reason_code")
print("ok")
PY
}

as_of_date=$(python3 - <<'PY'
import datetime as dt
print(dt.datetime.now(dt.timezone.utc).date().isoformat())
PY
)
from_date=$(python3 - <<'PY'
import datetime as dt
print((dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=5)).isoformat())
PY
)
start_ts=$(python3 - <<'PY'
import datetime as dt
end = dt.datetime.now(dt.timezone.utc)
start = end - dt.timedelta(hours=6)
print(start.isoformat().replace('+00:00', 'Z'))
PY
)
end_ts=$(python3 - <<'PY'
import datetime as dt
end = dt.datetime.now(dt.timezone.utc)
print(end.isoformat().replace('+00:00', 'Z'))
PY
)

symbols_json='{"mode":"default"}'

echo "== FTIP Milestone B+C+D Verification =="

echo "-- Public endpoints --"
health_file="${tmpdir}/health.json"
HTTP_CODE=$(curl_json GET "/health" "$health_file" 30 "")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$health_file"
  fail "/health HTTP ${HTTP_CODE}"
fi
pass "/health"

version_file="${tmpdir}/version.json"
HTTP_CODE=$(curl_json GET "/version" "$version_file" 30 "")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$version_file"
  fail "/version HTTP ${HTTP_CODE}"
fi
pass "/version"

echo "-- Protected endpoints --"
universe_file="${tmpdir}/universe.json"
HTTP_CODE=$(curl -s -o "$universe_file" -w "%{http_code}" --max-time 30 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$symbols_json" "$BASE/jobs/data/universe")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$universe_file"
  fail "/jobs/data/universe HTTP ${HTTP_CODE}"
fi
pass "/jobs/data/universe"

bars_daily_file="${tmpdir}/bars_daily.json"
bars_daily_payload=$(python3 - <<'PY' "$from_date" "$as_of_date"
import json, sys
payload = {
    "from_date": sys.argv[1],
    "to_date": sys.argv[2],
    "as_of_date": sys.argv[2],
}
print(json.dumps(payload))
PY
)
HTTP_CODE=$(curl -s -o "$bars_daily_file" -w "%{http_code}" --max-time 240 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$bars_daily_payload" "$BASE/jobs/data/bars-daily")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$bars_daily_file"
  fail "/jobs/data/bars-daily HTTP ${HTTP_CODE}"
fi
if [ -z "$(parse_field "$bars_daily_file" run_id)" ]; then
  out_json "$bars_daily_file"
  fail "bars-daily missing run_id"
fi
if ! ensure_failed_reason_codes "$bars_daily_file" >/dev/null; then
  out_json "$bars_daily_file"
  fail "bars-daily missing reason_code"
fi
pass "/jobs/data/bars-daily"

bars_intraday_file="${tmpdir}/bars_intraday.json"
bars_intraday_payload=$(python3 - <<'PY' "$start_ts" "$end_ts"
import json, sys
payload = {
    "start_ts": sys.argv[1],
    "end_ts": sys.argv[2],
    "timeframe": "5m",
}
print(json.dumps(payload))
PY
)
HTTP_CODE=$(curl -s -o "$bars_intraday_file" -w "%{http_code}" --max-time 240 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$bars_intraday_payload" "$BASE/jobs/data/bars-intraday")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$bars_intraday_file"
  fail "/jobs/data/bars-intraday HTTP ${HTTP_CODE}"
fi
if ! ensure_failed_reason_codes "$bars_intraday_file" >/dev/null; then
  out_json "$bars_intraday_file"
  fail "bars-intraday missing reason_code"
fi
pass "/jobs/data/bars-intraday"

news_file="${tmpdir}/news.json"
news_payload=$(python3 - <<'PY' "$start_ts" "$end_ts"
import json, sys
payload = {
    "from_ts": sys.argv[1],
    "to_ts": sys.argv[2],
}
print(json.dumps(payload))
PY
)
HTTP_CODE=$(curl -s -o "$news_file" -w "%{http_code}" --max-time 240 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$news_payload" "$BASE/jobs/data/news")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$news_file"
  fail "/jobs/data/news HTTP ${HTTP_CODE}"
fi
if ! ensure_failed_reason_codes "$news_file" >/dev/null; then
  out_json "$news_file"
  fail "news missing reason_code"
fi
pass "/jobs/data/news"

sentiment_file="${tmpdir}/sentiment.json"
sentiment_payload=$(python3 - <<'PY' "$as_of_date"
import json, sys
payload = {"as_of_date": sys.argv[1]}
print(json.dumps(payload))
PY
)
HTTP_CODE=$(curl -s -o "$sentiment_file" -w "%{http_code}" --max-time 240 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$sentiment_payload" "$BASE/jobs/data/sentiment-daily")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$sentiment_file"
  fail "/jobs/data/sentiment-daily HTTP ${HTTP_CODE}"
fi
pass "/jobs/data/sentiment-daily"

features_file="${tmpdir}/features.json"
features_payload=$(python3 - <<'PY' "$as_of_date"
import json, sys
payload = {"as_of_date": sys.argv[1], "lookback_days": 120}
print(json.dumps(payload))
PY
)
HTTP_CODE=$(curl -s -o "$features_file" -w "%{http_code}" --max-time 240 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$features_payload" "$BASE/jobs/features/daily")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$features_file"
  fail "/jobs/features/daily HTTP ${HTTP_CODE}"
fi
pass "/jobs/features/daily"

signals_file="${tmpdir}/signals.json"
signals_payload=$(python3 - <<'PY' "$as_of_date"
import json, sys
payload = {"as_of_date": sys.argv[1]}
print(json.dumps(payload))
PY
)
HTTP_CODE=$(curl -s -o "$signals_file" -w "%{http_code}" --max-time 240 -X POST -H "$AUTH_HEADER" -H "Content-Type: application/json" -d "$signals_payload" "$BASE/jobs/signals/daily")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$signals_file"
  fail "/jobs/signals/daily HTTP ${HTTP_CODE}"
fi
pass "/jobs/signals/daily"

latest_file="${tmpdir}/latest.json"
HTTP_CODE=$(curl_json GET "/signals/latest?symbol=AAPL" "$latest_file" 30 "$AUTH_HEADER")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$latest_file"
  fail "/signals/latest HTTP ${HTTP_CODE}"
fi
if [ -z "$(parse_field "$latest_file" action)" ]; then
  out_json "$latest_file"
  fail "/signals/latest missing action"
fi
pass "/signals/latest"

top_file="${tmpdir}/top.json"
HTTP_CODE=$(curl_json GET "/signals/top?mode=buy&limit=5&country=ALL" "$top_file" 30 "$AUTH_HEADER")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$top_file"
  fail "/signals/top HTTP ${HTTP_CODE}"
fi
python3 - <<'PY' "$top_file"
import json, sys
payload = json.load(open(sys.argv[1]))
if not isinstance(payload, list) or len(payload) < 1:
    sys.exit("expected at least one entry")
PY
pass "/signals/top"

evidence_file="${tmpdir}/evidence.json"
HTTP_CODE=$(curl_json GET "/signals/evidence?symbol=AAPL&as_of_date=${as_of_date}" "$evidence_file" 30 "$AUTH_HEADER")
if [ "$HTTP_CODE" != "200" ]; then
  out_json "$evidence_file"
  fail "/signals/evidence HTTP ${HTTP_CODE}"
fi
if [ -z "$(parse_field "$evidence_file" signal.action)" ]; then
  out_json "$evidence_file"
  fail "/signals/evidence missing signal"
fi
pass "/signals/evidence"

pass "milestone B+C+D verification complete"
