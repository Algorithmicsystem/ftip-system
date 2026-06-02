"""Phase 2.3: Cross-Sector Linkage Intelligence API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

router = APIRouter(prefix="/linkage", tags=["linkage"])


@router.get("/peers/{symbol}")
def get_peers(
    symbol: str,
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return active sector peers for a symbol with their current AXIOM scores."""
    from api.jobs.regime_analogs import get_peers_with_axiom
    date = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    peers = get_peers_with_axiom(symbol.upper(), date)
    return {
        "symbol": symbol.upper(),
        "as_of_date": date.isoformat(),
        "count": len(peers),
        "peers": peers,
    }


@router.get("/stress-propagation/{symbol}")
def get_stress_propagation(
    symbol: str,
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Show how fragility stress in this symbol propagates to linked peers."""
    from api.jobs.regime_analogs import get_stress_propagation
    date = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    return get_stress_propagation(symbol.upper(), date)
