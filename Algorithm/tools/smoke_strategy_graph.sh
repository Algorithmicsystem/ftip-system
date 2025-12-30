#!/usr/bin/env bash
# Smoke test for Strategy Graph Engine endpoints.
# If using zsh, run `setopt interactivecomments` before executing to avoid '#' parsing issues.

set -euo pipefail

BASE_URL=${BASE_URL:-"http://localhost:8000"}
SYMBOL=${1:-AAPL}
AS_OF=${2:-"2024-01-10"}
LOOKBACK=${LOOKBACK:-252}

printf "Running strategy graph smoke for %s as_of=%s\n" "$SYMBOL" "$AS_OF"

curl -s -X POST "${BASE_URL}/prosperity/strategy_graph/run" \
  -H "Content-Type: application/json" \
  -d "{\"symbols\":[\"${SYMBOL}\"],\"from_date\":\"${AS_OF}\",\"to_date\":\"${AS_OF}\",\"as_of_date\":\"${AS_OF}\",\"lookback\":${LOOKBACK}}" | jq . || true

curl -s "${BASE_URL}/prosperity/strategy_graph/latest/ensemble?symbol=${SYMBOL}&lookback=${LOOKBACK}" | jq . || true
curl -s "${BASE_URL}/prosperity/strategy_graph/latest/strategies?symbol=${SYMBOL}&lookback=${LOOKBACK}" | jq . || true
