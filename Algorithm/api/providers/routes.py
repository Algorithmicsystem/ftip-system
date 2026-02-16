from __future__ import annotations

from fastapi import APIRouter

from api.providers import ProvidersHealthResponse, get_providers_health

router = APIRouter()


@router.get("/providers/health", response_model=ProvidersHealthResponse)
def providers_health() -> ProvidersHealthResponse:
    return get_providers_health()
