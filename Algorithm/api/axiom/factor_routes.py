"""GET /axiom/factors/{symbol} — Factor model decomposition for a symbol."""
from __future__ import annotations
import json
import datetime as dt
import statistics
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query
from api import db, security

router = APIRouter(
    prefix="/axiom",
    tags=["axiom"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)


@router.get("/factors/{symbol}")
def get_factor_loadings(
    symbol: str,
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    date = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    symbol = symbol.upper()

    if not db.db_read_enabled():
        return _empty_response(symbol, date)

    row = db.safe_fetchone(
        """
        SELECT payload FROM axiom_scores_daily
        WHERE symbol = %s AND as_of_date <= %s
        ORDER BY as_of_date DESC LIMIT 1
        """,
        (symbol, date),
    )
    if not row:
        return _empty_response(symbol, date)

    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")

    factor_loadings_raw = payload.get("factor_loadings_summary") or []
    alpha_decomp = payload.get("alpha_decomposition") or {}
    fcs = payload.get("factor_composite_score") or 50.0
    regime = str(payload.get("regime_label") or "CHOPPY")

    # Dominant factors: top 3 by |loading|
    sorted_factors = sorted(factor_loadings_raw, key=lambda f: abs(f.get("loading") or 0), reverse=True)
    dominant_factors = sorted_factors[:3]

    # Regime sensitivity: compute weighted loading across all four regimes
    from api.axiom.factors.regime_factor_matrix import FACTOR_REGIME_MATRIX
    regime_scores = {}
    for reg, weights in FACTOR_REGIME_MATRIX.items():
        reg_score = sum(
            (fl.get("loading") or 0) * weights.get(fl.get("factor_name", ""), 0)
            for fl in factor_loadings_raw
        )
        regime_scores[reg] = round(reg_score * 50.0 + 50.0, 2)

    best_regime = max(regime_scores, key=regime_scores.get) if regime_scores else regime
    worst_regime = min(regime_scores, key=regime_scores.get) if regime_scores else regime
    sensitivity_score = round(
        min(statistics.stdev(regime_scores.values()) / 50.0 * 100.0, 100.0)
        if len(regime_scores) > 1 else 0.0, 2
    )

    return {
        "symbol": symbol,
        "as_of_date": date.isoformat(),
        "factor_composite_score": fcs,
        "factor_loadings": factor_loadings_raw,
        "alpha_decomposition": alpha_decomp,
        "dominant_factors": dominant_factors,
        "regime_sensitivity": {
            "most_favorable_regime": best_regime,
            "least_favorable_regime": worst_regime,
            "regime_sensitivity_score": sensitivity_score,
        },
    }


def _empty_response(symbol: str, date: dt.date) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "as_of_date": date.isoformat(),
        "factor_composite_score": 50.0,
        "factor_loadings": [],
        "alpha_decomposition": {},
        "dominant_factors": [],
        "regime_sensitivity": {
            "most_favorable_regime": None,
            "least_favorable_regime": None,
            "regime_sensitivity_score": 0.0,
        },
    }
