#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:8000}
KEY=${KEY:-demo-key}

print_head() {
  local label="$1"
  local cmd="$2"
  echo "${label}" && eval "${cmd} | head"
}

echo "1) Core health/version endpoints"
for path in /health /version; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}${path}")
  echo "   ${path} -> HTTP ${code}"
done

echo "\n2) Prosperity health should be protected (no key)"
unauth_status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/prosperity/health")
echo "   /prosperity/health (no key) -> HTTP ${unauth_status}"

echo "\n3) Prosperity health with API key"
auth_status=$(curl -s -o /dev/null -w "%{http_code}" -H "X-FTIP-API-Key: ${KEY}" "${BASE}/prosperity/health")
echo "   /prosperity/health (with key) -> HTTP ${auth_status}"

payload='{"symbols":["AAPL","MSFT","NVDA","AMZN","TSLA"],"from_date":"2024-01-01","to_date":"2024-01-05","as_of_date":"2024-01-05","lookback":252}'

echo "\n4) Run snapshot for five symbols"
curl -s -X POST "${BASE}/prosperity/snapshot/run" \
  -H "Content-Type: application/json" \
  -H "X-FTIP-API-Key: ${KEY}" \
  -d "${payload}" | head

echo "\n5) Latest signal for AAPL"
print_head "   latest signal" "curl -s -H 'X-FTIP-API-Key: ${KEY}' '${BASE}/prosperity/latest/signal?symbol=AAPL&lookback=252'"

echo "\n6) Strategy graph sample for AAPL"
print_head "   strategy graph" "curl -s -H 'X-FTIP-API-Key: ${KEY}' '${BASE}/prosperity/graph/strategy?symbol=AAPL&lookback=252&days=365'"
