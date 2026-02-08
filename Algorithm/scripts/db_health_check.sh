#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
tmpdir=$(mktemp -d)
out_json() { cat "$1"; }
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; exit 1; }

health_file="${tmpdir}/db_health.json"
status=$(curl -s -o "$health_file" -w "%{http_code}" "${BASE}/db/health")

if [[ "$status" != "200" ]]; then
  out_json "$health_file"
  fail "/db/health HTTP ${status}"
fi

python - "$health_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
if payload.get("status") == "disabled":
    print("DB disabled; skipping schema version check.")
    sys.exit(0)

latest = payload.get("latest_migration")
if not latest:
    print("Missing latest_migration in /db/health response.")
    sys.exit(1)
print(f"Latest migration: {latest}")
PY
pass "/db/health schema version"

rm -rf "$tmpdir"
