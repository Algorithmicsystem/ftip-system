"""Backward-compatibility shim — routes are now in api/narrator/routes.py."""
from __future__ import annotations

from fastapi import APIRouter

from api.narrator.routes import (  # noqa: F401
    _market_stats_from_features,
    _performance_defaults,
    _prepare_citations,
    _resolve_signal,
    _safe_perf_value,
    narrator_client as client,
)

# Empty router retained so existing imports of `router` don't break.
router = APIRouter()

__all__ = ["router", "client", "_resolve_signal"]
