#!/usr/bin/env sh
set -eu

APP_DIR="/app"
cd "$APP_DIR"

PORT="${PORT:-8000}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

DB_ENABLED="${FTIP_DB_ENABLED:-}"
MIGRATIONS_AUTO="${FTIP_MIGRATIONS_AUTO:-}"
DATABASE_URL="${DATABASE_URL:-}"
DB_REQUIRED="${FTIP_DB_REQUIRED:-}"

if is_truthy "$DB_ENABLED"; then
  if [ -z "$DB_REQUIRED" ]; then
    export FTIP_DB_REQUIRED=1
    echo "FTIP_DB_REQUIRED not set; defaulting to 1 for DB-backed runtime safety."
  fi
  if [ -z "$DATABASE_URL" ]; then
    echo "DATABASE_URL is required when FTIP_DB_ENABLED is true." >&2
    exit 1
  fi
  if is_truthy "$MIGRATIONS_AUTO"; then
    echo "Running migrations..."
    python -c "from api import db; db.apply_migrations()"
  else
    echo "Migrations disabled; skipping (ensure POST /prosperity/bootstrap ran successfully)."
  fi
else
  echo "Database disabled; skipping migrations."
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
