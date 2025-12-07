#!/usr/bin/env bash
set -e

cd Algorithm

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Run FastAPI via Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
