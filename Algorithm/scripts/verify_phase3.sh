#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
API_KEY=${API_KEY:-demo-key}
PROTECTED_PATH=${PROTECTED_PATH:-/prosperity/narrator/health}

echo "1) Public health endpoints (no API key)"
for path in /health /version /db/health /prosperity/health; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${path}")
  echo "   ${path} -> HTTP ${status}"
done

echo "\n2) Protected prosperity endpoint should return 401 without key"
unauth_status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${PROTECTED_PATH}")
echo "   ${PROTECTED_PATH} (no key) -> HTTP ${unauth_status}"

echo "\n3) Protected endpoint with key"
auth_status=$(curl -s -o /dev/null -w "%{http_code}" -H "X-FTIP-API-Key: ${API_KEY}" "${BASE_URL}${PROTECTED_PATH}")
echo "   ${PROTECTED_PATH} (with key) -> HTTP ${auth_status}"

echo "\n4) Demonstrate rate limiting (will stop after first 429)"
for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "X-FTIP-API-Key: ${API_KEY}" "${BASE_URL}${PROTECTED_PATH}")
  echo "   hit ${i} -> HTTP ${code}"
  if [[ "${code}" == "429" ]]; then
    break
  fi
  sleep 0.05
done

echo "\n5) CORS/domain readiness"
curl -s "${BASE_URL}/ops/domain" | sed 's/{/\n{/'
