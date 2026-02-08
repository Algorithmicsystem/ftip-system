SHELL := /bin/bash

COMPOSE := docker-compose
ENV_FILE ?= .env
RUN_ENV = if [ -f $(ENV_FILE) ]; then set -a; . $(ENV_FILE); set +a; fi;

.PHONY: help dev db-up db-down db-reset migrate db-check test lint fmt

help:
	@echo "Targets:"
	@echo "  make dev       Start Postgres, run migrations, then start the API (uvicorn)"
	@echo "  make db-up     Start Postgres via docker compose"
	@echo "  make db-down   Stop Postgres via docker compose"
	@echo "  make db-reset  Drop/recreate schema and re-run migrations"
	@echo "  make migrate   Run database migrations"
	@echo "  make db-check  Verify database connectivity"
	@echo "  make test      Run pytest"
	@echo "  make lint      Run ruff checks"
	@echo "  make fmt       Run ruff fixes and black formatting"

db-up:
	@$(COMPOSE) up -d db


db-down:
	@$(COMPOSE) down -v


db-check:
	@$(RUN_ENV) python tools/db_check.py --retries 10 --sleep 1


migrate:
	@$(RUN_ENV) python tools/migrate_db.py


db-reset:
	@$(RUN_ENV) python tools/reset_db.py


dev: db-up migrate db-check
	@$(RUN_ENV) uvicorn api.main:app --reload --port $${PORT:-8000}


test:
	@$(RUN_ENV) pytest -q


lint:
	@ruff check .


fmt:
	@ruff check --fix .
	@black .
