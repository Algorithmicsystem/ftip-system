#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${FTIP_BASE_URL:-http://localhost:8000}
CURL=${CURL:-curl}

function log() {
  echo "[smoke] $1"
}

function get() {
  local path="$1"
  log "GET ${BASE_URL}${path}"
  ${CURL} -sf "${BASE_URL}${path}"
}

function post_json() {
  local path="$1"
  local payload="$2"
  log "POST ${BASE_URL}${path}"
  ${CURL} -sf -H "Content-Type: application/json" -d "${payload}" "${BASE_URL}${path}"
}

log "starting prosperity smoke"
get "/health"
get "/db/health"
get "/prosperity/health"

post_json "/prosperity/snapshot/run" '{"symbols":["AAPL","MSFT","GOOG","AMZN","META"],"from_date":"2024-01-02","to_date":"2024-02-02","as_of_date":"2024-02-02","lookback":63,"concurrency":3}'

get "/prosperity/latest/signal?symbol=AAPL&lookback=63"
get "/prosperity/latest/features?symbol=AAPL&lookback=63"

log "prosperity smoke completed"
