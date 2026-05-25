"""Session 19: Portfolio risk overlay.

POST /jobs/risk/overlay — compute concentration, sector exposure,
regime analysis, regime-flip detection, and overall risk state for a
given portfolio of symbol/weight positions.

No new DB table — reads from axiom_scores_daily, market_symbols,
signal_ic_daily, and market_breadth_daily.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

VETO_REGIMES = frozenset({"euphoria_critical", "liquidity_fracture"})
FAVORABLE_REGIMES = frozenset({
    "fundamental_convergence", "compensation_capture",
    "behavioral_continuation", "convexity_opportunity", "recovery_reset",
})

_HIGH_FRAGILITY_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

def compute_concentration(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = sorted(
        (float(p.get("weight", 0.0)) for p in positions if float(p.get("weight", 0.0)) > 0),
        reverse=True,
    )
    if not weights:
        return {
            "hhi": 0.0, "effective_n": 0.0,
            "top3_weight": 0.0, "max_single_weight": 0.0,
            "concentration_state": "UNKNOWN",
        }

    hhi = round(sum(w * w for w in weights), 4)
    effective_n = round(1.0 / hhi, 2) if hhi > 0 else 0.0
    top3 = round(sum(weights[:3]), 4)

    if hhi > 0.33:
        state = "HIGH"
    elif hhi > 0.20:
        state = "MODERATE"
    elif hhi > 0.10:
        state = "MILD"
    else:
        state = "LOW"

    return {
        "hhi": hhi,
        "effective_n": effective_n,
        "top3_weight": top3,
        "max_single_weight": round(weights[0], 4),
        "concentration_state": state,
    }


# ---------------------------------------------------------------------------
# Sector exposure
# ---------------------------------------------------------------------------

def _load_sector_map(symbols: List[str]) -> Dict[str, str]:
    if not db.db_read_enabled() or not symbols:
        return {}
    try:
        rows = db.safe_fetchall(
            "SELECT symbol, sector FROM market_symbols WHERE symbol = ANY(%s) AND sector IS NOT NULL",
            (symbols,),
        )
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def compute_sector_exposure(
    positions: List[Dict[str, Any]],
    sector_map: Dict[str, str],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for p in positions:
        sym = p.get("symbol", "")
        w = float(p.get("weight", 0.0))
        sector = sector_map.get(sym, "unknown")
        totals[sector] = round(totals.get(sector, 0.0) + w, 4)
    return dict(sorted(totals.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# AXIOM enrichment
# ---------------------------------------------------------------------------

def _load_axiom_data(
    symbols: List[str],
    as_of_date: dt.date,
) -> Dict[str, Dict[str, Any]]:
    if not db.db_read_enabled() or not symbols:
        return {}
    try:
        rows = db.safe_fetchall(
            """
            SELECT
                symbol,
                regime_label,
                deployable_alpha_utility,
                deployability_tier,
                overall_confidence,
                (payload->'engine_scores'->'critical_fragility'->>'score')::numeric AS fragility
            FROM axiom_scores_daily
            WHERE symbol = ANY(%s) AND as_of_date = %s
            """,
            (symbols, as_of_date),
        )
        result: Dict[str, Dict[str, Any]] = {}
        for sym, regime, dau, tier, conf, frag in rows:
            result[sym] = {
                "regime_label": regime,
                "dau": float(dau) if dau is not None else None,
                "deployability_tier": tier,
                "overall_confidence": float(conf) if conf is not None else None,
                "fragility_score": float(frag) if frag is not None else None,
            }
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# IC / breadth state helpers (inlined to keep module self-contained)
# ---------------------------------------------------------------------------

def _load_ic_state(as_of_date: dt.date) -> str:
    if not db.db_read_enabled():
        return "INSUFFICIENT"
    try:
        row = db.safe_fetchone(
            """
            SELECT ic_state FROM signal_ic_daily
            WHERE score_field = 'composite' AND horizon_label = '21d'
              AND as_of_date <= %s
            ORDER BY as_of_date DESC LIMIT 1
            """,
            (as_of_date,),
        )
        return str(row[0]) if row and row[0] else "INSUFFICIENT"
    except Exception:
        return "INSUFFICIENT"


def _load_breadth_state(as_of_date: dt.date) -> str:
    if not db.db_read_enabled():
        return "NEUTRAL"
    try:
        row = db.safe_fetchone(
            "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s",
            (as_of_date,),
        )
        return str(row[0]) if row and row[0] else "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# Regime flip detection
# ---------------------------------------------------------------------------

def detect_regime_flips(
    symbols: List[str],
    as_of_date: dt.date,
    lookback_days: int = 30,
) -> List[Dict[str, Any]]:
    if not db.db_read_enabled() or not symbols:
        return []
    try:
        current_rows = db.safe_fetchall(
            "SELECT symbol, regime_label FROM axiom_scores_daily WHERE symbol = ANY(%s) AND as_of_date = %s",
            (symbols, as_of_date),
        )
        current_map = {r[0]: r[1] for r in current_rows}

        prev_rows = db.safe_fetchall(
            """
            SELECT DISTINCT ON (symbol) symbol, regime_label, as_of_date
            FROM axiom_scores_daily
            WHERE symbol = ANY(%s)
              AND as_of_date < %s
              AND as_of_date >= %s
            ORDER BY symbol, as_of_date DESC
            """,
            (symbols, as_of_date, as_of_date - dt.timedelta(days=lookback_days)),
        )
        prev_map = {r[0]: (r[1], r[2]) for r in prev_rows}

        flips: List[Dict[str, Any]] = []
        for sym, curr_regime in current_map.items():
            if sym not in prev_map:
                continue
            prev_regime, prev_date = prev_map[sym]
            if prev_regime and curr_regime and prev_regime != curr_regime:
                days_ago = (as_of_date - prev_date).days if hasattr(prev_date, "year") else None
                flips.append({
                    "symbol": sym,
                    "prev_regime": prev_regime,
                    "curr_regime": curr_regime,
                    "days_ago": days_ago,
                })
        return flips
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def _risk_score_and_flags(
    concentration: Dict[str, Any],
    sector_exposure: Dict[str, float],
    axiom_data: Dict[str, Dict[str, Any]],
    positions: List[Dict[str, Any]],
    regime_flips: List[Dict[str, Any]],
    ic_state: str,
) -> Tuple[float, str, List[Dict[str, Any]]]:
    penalty = 0.0
    flags: List[Dict[str, Any]] = []

    # 1. Concentration
    hhi = float(concentration.get("hhi", 0.0))
    if hhi > 0.33:
        penalty += 25
    elif hhi > 0.20:
        penalty += 15
    elif hhi > 0.10:
        penalty += 8

    # 2. Veto regime exposure
    veto_weight = 0.0
    for p in positions:
        sym = p.get("symbol", "")
        w = float(p.get("weight", 0.0))
        regime = (axiom_data.get(sym) or {}).get("regime_label") or ""
        if regime in VETO_REGIMES:
            veto_weight += w
            flags.append({"symbol": sym, "flag": "veto_regime", "detail": regime, "weight": round(w, 4)})
    penalty += min(veto_weight * 150, 30)

    # 3. IC state
    ic_penalties = {"DEGRADED": 15, "WEAK": 8, "INSUFFICIENT": 5}
    penalty += ic_penalties.get(ic_state or "", 0)
    if ic_state == "DEGRADED":
        flags.append({"symbol": None, "flag": "ic_degraded", "detail": "IC state is DEGRADED", "weight": None})

    # 4. Sector concentration
    max_sector = max(sector_exposure.values(), default=0.0)
    if max_sector > 0.60:
        penalty += 15
    elif max_sector > 0.40:
        penalty += 8
    if max_sector > 0.40:
        top_sector = max(sector_exposure, key=sector_exposure.__getitem__)
        flags.append({
            "symbol": None,
            "flag": "sector_concentration",
            "detail": f"{top_sector}: {round(max_sector * 100, 1)}%",
            "weight": round(max_sector, 4),
        })

    # 5. Regime flips
    if len(regime_flips) >= 3:
        penalty += 15
    elif regime_flips:
        penalty += 8

    # 6. High fragility (informational — no penalty)
    for p in positions:
        sym = p.get("symbol", "")
        frag = (axiom_data.get(sym) or {}).get("fragility_score")
        if frag is not None and frag >= _HIGH_FRAGILITY_THRESHOLD:
            flags.append({
                "symbol": sym,
                "flag": "high_fragility",
                "detail": f"fragility={round(frag, 1)}",
                "weight": round(float(p.get("weight", 0.0)), 4),
            })

    score = round(min(penalty, 100.0), 1)
    if score >= 65:
        state = "CRITICAL"
    elif score >= 40:
        state = "HIGH"
    elif score >= 20:
        state = "MODERATE"
    else:
        state = "LOW"

    return score, state, flags


# ---------------------------------------------------------------------------
# Main overlay function
# ---------------------------------------------------------------------------

def compute_portfolio_risk(
    as_of_date: dt.date,
    portfolio: List[Dict[str, Any]],
    *,
    flip_lookback_days: int = 30,
) -> Dict[str, Any]:
    positions = [
        {
            "symbol": (p.get("symbol") or "").strip().upper(),
            "weight": float(p.get("weight", 0.0)),
        }
        for p in portfolio
        if (p.get("symbol") or "").strip()
    ]
    symbols = [p["symbol"] for p in positions]
    gross_weight = round(sum(p["weight"] for p in positions), 4)

    sector_map  = _load_sector_map(symbols)
    axiom_data  = _load_axiom_data(symbols, as_of_date)
    flips       = detect_regime_flips(symbols, as_of_date, flip_lookback_days)
    ic_state    = _load_ic_state(as_of_date)
    breadth_state = _load_breadth_state(as_of_date)

    concentration   = compute_concentration(positions)
    sector_exposure = compute_sector_exposure(positions, sector_map)

    regime_breakdown: Dict[str, int] = {}
    for p in positions:
        regime = (axiom_data.get(p["symbol"]) or {}).get("regime_label") or "unknown"
        regime_breakdown[regime] = regime_breakdown.get(regime, 0) + 1

    risk_score, risk_state, flags = _risk_score_and_flags(
        concentration, sector_exposure, axiom_data, positions, flips, ic_state,
    )

    max_sect = max(sector_exposure.values(), default=0.0)
    if max_sect > 0.60:
        sector_conc_state = "HIGH"
    elif max_sect > 0.40:
        sector_conc_state = "MODERATE"
    else:
        sector_conc_state = "LOW"

    per_symbol = [
        {
            "symbol": p["symbol"],
            "weight": p["weight"],
            "regime_label": (axiom_data.get(p["symbol"]) or {}).get("regime_label") or "unknown",
            "regime_aligned": ((axiom_data.get(p["symbol"]) or {}).get("regime_label") or "") in FAVORABLE_REGIMES,
            "regime_veto": ((axiom_data.get(p["symbol"]) or {}).get("regime_label") or "") in VETO_REGIMES,
            "dau": (axiom_data.get(p["symbol"]) or {}).get("dau"),
            "deployability_tier": (axiom_data.get(p["symbol"]) or {}).get("deployability_tier"),
            "fragility_score": (axiom_data.get(p["symbol"]) or {}).get("fragility_score"),
            "flags": [f["flag"] for f in flags if f.get("symbol") == p["symbol"]],
        }
        for p in positions
    ]

    logger.info(
        "risk.overlay as_of=%s symbols=%d risk_score=%.1f risk_state=%s flags=%d",
        as_of_date, len(symbols), risk_score, risk_state, len(flags),
    )

    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "portfolio_size": len(positions),
        "gross_weight": gross_weight,
        "concentration": concentration,
        "sector_exposure": sector_exposure,
        "sector_concentration_state": sector_conc_state,
        "regime_breakdown": regime_breakdown,
        "regime_flips": flips,
        "risk_flags": flags,
        "ic_state": ic_state,
        "breadth_state": breadth_state,
        "risk_score": risk_score,
        "risk_state": risk_state,
        "per_symbol": per_symbol,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class PortfolioPosition(BaseModel):
    symbol: str
    weight: float = Field(ge=0.0, le=1.0)


class RiskOverlayRequest(BaseModel):
    portfolio: List[PortfolioPosition]
    as_of_date: Optional[str] = None
    flip_lookback_days: int = Field(default=30, ge=1, le=365)


def _resolve_date(raw: Optional[str]) -> dt.date:
    if raw:
        try:
            return dt.date.fromisoformat(raw)
        except ValueError:
            pass
    return dt.date.today() - dt.timedelta(days=1)


@router.post("/risk/overlay")
def risk_overlay(req: RiskOverlayRequest) -> Dict[str, Any]:
    as_of = _resolve_date(req.as_of_date)
    portfolio = [{"symbol": p.symbol, "weight": p.weight} for p in req.portfolio]
    return compute_portfolio_risk(as_of, portfolio, flip_lookback_days=req.flip_lookback_days)
