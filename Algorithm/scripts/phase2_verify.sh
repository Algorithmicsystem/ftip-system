#!/usr/bin/env bash
set -e

BASE="${BASE:-https://ftip-system-production.up.railway.app}"
KEY="${KEY:-}"

echo "== health/version =="
curl -s "$BASE/health" ; echo
curl -s "$BASE/version" ; echo

echo
echo "== prosperity health (no key, should be 401) =="
curl -s -i "$BASE/prosperity/health" | head -n 5 ; echo

echo
echo "== prosperity health (with key, should be 200) =="
curl -s -H "X-FTIP-API-Key: $KEY" "$BASE/prosperity/health" ; echo

echo
echo "== run snapshot =="
curl -s -X POST "$BASE/prosperity/snapshot/run" \
  -H "Content-Type: application/json" \
  -H "X-FTIP-API-Key: $KEY" \
  -d '{
    "symbols":["AAPL","MSFT","NVDA","AMZN","TSLA"],
    "from_date":"2023-01-01",
    "to_date":"2024-12-31",
    "as_of_date":"2024-12-31",
    "lookback":252,
    "concurrency":3
  }' | head -c 1200 ; echo

echo
echo "== latest signal =="
curl -s -H "X-FTIP-API-Key: $KEY" \
  "$BASE/prosperity/latest/signal?symbol=AAPL&lookback=252" | head -c 1200 ; echo

echo
echo "== strategy graph =="
curl -s -H "X-FTIP-API-Key: $KEY" \
  "$BASE/prosperity/graph/strategy?symbol=AAPL&lookback=252&days=365" | head -c 1200 ; echo

echo
echo "== universe graph =="
curl -s -H "X-FTIP-API-Key: $KEY" \
  "$BASE/prosperity/graph/universe" | head -c 1200 ; echo

