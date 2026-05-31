from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from api.providers import ProvidersHealthResponse, get_providers_health
from api.providers.reliability import (
    get_provider_reliability_summary,
    get_provider_reliability_window,
    snapshot_provider_reliability,
)

router = APIRouter()


@router.get("/providers/health", response_model=ProvidersHealthResponse)
def providers_health() -> ProvidersHealthResponse:
    health = get_providers_health()
    # Persist today's snapshot as a side-effect (best-effort, never raises).
    try:
        snapshot_provider_reliability(health)
    except Exception:  # pragma: no cover
        pass
    return health


@router.get("/providers/reliability")
def providers_reliability(
    days: int = Query(30, ge=1, le=365),
    provider: Optional[str] = Query(None),
):
    """Provider reliability time-series for the last N days.

    Returns summary (uptime_pct per provider) plus raw daily records.
    """
    summary = get_provider_reliability_summary(days=days)
    records = get_provider_reliability_window(days=days, provider=provider or None)
    return {
        "window_days": days,
        "summary": summary,
        "records": [r.model_dump() for r in records],
    }
