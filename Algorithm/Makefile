SHELL := /bin/bash

COMPOSE := docker-compose
ENV_FILE ?= .env
VENV_PY := .venv/bin/python
RUN_ENV = if [ -f $(ENV_FILE) ]; then set -a; . $(ENV_FILE); set +a; fi;

.PHONY: help dev compose-up-db compose-down-db db-up db-down db-reset migrate db-check env-check test lint fmt

help:
	@echo "Targets:"
	@echo "  make dev            Start the API (uvicorn)"
	@echo "  make compose-up-db  Start Postgres via docker-compose"
	@echo "  make compose-down-db Stop Postgres via docker-compose"
	@echo "  make db-reset  Drop/recreate schema and re-run migrations"
	@echo "  make migrate   Run database migrations"
	@echo "  make db-check  Verify database connectivity"
	@echo "  make env-check Print FTIP_DB_ENABLED and DATABASE_URL"
	@echo "  make test      Run pytest"
	@echo "  make lint      Run ruff checks"
	@echo "  make fmt       Run ruff fixes and black formatting"

define require_venv
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "Missing .venv. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"; \
		exit 1; \
	fi
endef


compose-up-db:
	@$(COMPOSE) up -d db


compose-down-db:
	@$(COMPOSE) down -v


db-up: compose-up-db


db-down: compose-down-db


env-check:
	@$(call require_venv)
	@$(RUN_ENV) $(VENV_PY) tools/env_check.py

db-check:
	@$(call require_venv)
	@$(RUN_ENV) $(VENV_PY) tools/db_check.py --retries 10 --sleep 1


migrate:
	@$(call require_venv)
	@$(RUN_ENV) $(VENV_PY) tools/migrate_db.py


db-reset:
	@$(call require_venv)
	@$(RUN_ENV) $(VENV_PY) tools/reset_db.py


dev:
	@$(call require_venv)
	@$(RUN_ENV) $(VENV_PY) -m uvicorn api.main:app --reload --port $${PORT:-8000}


test:
	@$(call require_venv)
	@$(RUN_ENV) $(VENV_PY) -m pytest -q


lint:
	@$(call require_venv)
	@$(VENV_PY) -m ruff check .


fmt:
	@$(call require_venv)
	@$(VENV_PY) -m ruff check --fix .
	@$(VENV_PY) -m black .
