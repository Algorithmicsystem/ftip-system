"""Phase 18.5b: Cross-Asset and Global Macro Routes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/macro",
    tags=["macro"],
)


class MacroClassifyIn(BaseModel):
    gdp_growth: float
    cpi_yoy: float
    fed_funds_rate: float
    fed_funds_rate_1y_ago: float
    ig_credit_spread: float = 120.0


@router.get("/snapshot")
def get_macro_snapshot() -> Dict[str, Any]:
    from api.macro.cross_asset_engine import compute_cross_asset_snapshot
    from api.macro.global_macro import classify_macro_regime
    from api import db
    macro_meta: Dict[str, Any] = {}
    if db.db_read_enabled():
        try:
            row = db.safe_fetchone(
                "SELECT meta FROM macro_intelligence_snapshots ORDER BY as_of_date DESC LIMIT 1"
            )
            if row and row[0]:
                macro_meta = row[0] if isinstance(row[0], dict) else {}
        except Exception:
            pass

    snapshot = compute_cross_asset_snapshot({}, "UNKNOWN")
    macro = classify_macro_regime(
        gdp_growth=float(macro_meta.get("gdp_growth", 2.0)),
        cpi_yoy=float(macro_meta.get("cpi_yoy", 2.5)),
        fed_funds_rate=float(macro_meta.get("fed_funds_rate", 3.0)),
        fed_funds_rate_1y_ago=float(macro_meta.get("fed_funds_rate_1y_ago", 2.5)),
    )
    return {
        "cross_asset": {
            "as_of_date": snapshot.as_of_date.isoformat(),
            "equity_regime_confirmed": snapshot.equity_regime_confirmed,
            "cross_asset_confirmation_score": snapshot.cross_asset_confirmation_score,
            "fixed_income_signal": snapshot.fixed_income_signal,
            "currency_signal": snapshot.currency_signal,
            "commodity_signal": snapshot.commodity_signal,
            "volatility_signal": snapshot.volatility_signal,
            "equity_signal_amplifier": snapshot.equity_signal_amplifier,
            "macro_narrative": snapshot.macro_narrative,
        },
        "macro_intelligence": {
            "gdp_regime": macro.gdp_regime,
            "inflation_regime": macro.inflation_regime,
            "monetary_regime": macro.monetary_regime,
            "credit_regime": macro.credit_regime,
            "equity_macro_score": macro.equity_macro_score,
            "macro_environment_score": macro.macro_environment_score,
            "favored_factors": macro.favored_axiom_factors,
            "unfavored_factors": macro.unfavored_axiom_factors,
            "macro_regime_label": macro.macro_regime_label,
        },
    }


@router.get("/factor-overlay")
def get_factor_overlay() -> Dict[str, Any]:
    from api.macro.global_macro import classify_macro_regime, compute_macro_factor_overlay
    macro = classify_macro_regime(
        gdp_growth=2.0, cpi_yoy=2.5,
        fed_funds_rate=3.0, fed_funds_rate_1y_ago=2.5,
    )
    base_loadings = {f: 0.5 for f in ["EIF", "CMF", "BAF", "KLF", "SCAF", "ICF", "GBF", "MTRF", "MQF", "VIF", "RTF", "NTFF"]}
    adjusted = compute_macro_factor_overlay(macro, base_loadings)
    return {
        "macro_regime_label": macro.macro_regime_label,
        "favored_factors": macro.favored_axiom_factors,
        "unfavored_factors": macro.unfavored_axiom_factors,
        "adjusted_factor_weights": adjusted,
    }


@router.get("/equity-implications/{symbol}")
def get_equity_implications(symbol: str) -> Dict[str, Any]:
    from api import db
    from api.macro.cross_asset_engine import compute_cross_asset_for_equity
    payload: Dict[str, Any] = {}
    if db.db_read_enabled():
        try:
            row = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol=%s ORDER BY as_of_date DESC LIMIT 1",
                (symbol.upper(),),
            )
            payload = row[0] if row and isinstance(row[0], dict) else {}
        except Exception:
            pass
    return compute_cross_asset_for_equity(symbol.upper(), payload)


@router.get("/narrative")
def get_macro_narrative() -> Dict[str, Any]:
    from api.macro.cross_asset_engine import compute_cross_asset_snapshot
    from api.macro.global_macro import classify_macro_regime
    snapshot = compute_cross_asset_snapshot({}, "UNKNOWN")
    macro = classify_macro_regime(
        gdp_growth=2.0, cpi_yoy=2.5,
        fed_funds_rate=3.0, fed_funds_rate_1y_ago=2.5,
    )
    paragraph1 = snapshot.macro_narrative
    paragraph2 = macro.investment_implications
    paragraph3 = (
        f"Favored AXIOM factors in current environment: {', '.join(macro.favored_axiom_factors)}. "
        f"Unfavored factors: {', '.join(macro.unfavored_axiom_factors)}. "
        f"Macro environment score: {macro.macro_environment_score:.0f}/100."
    )
    return {
        "narrative": f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}",
        "macro_environment_score": macro.macro_environment_score,
        "cross_asset_confirmation": snapshot.cross_asset_confirmation_score,
    }


@router.post("/classify")
def classify_macro(body: MacroClassifyIn) -> Dict[str, Any]:
    from api.macro.global_macro import classify_macro_regime
    snap = classify_macro_regime(
        gdp_growth=body.gdp_growth,
        cpi_yoy=body.cpi_yoy,
        fed_funds_rate=body.fed_funds_rate,
        fed_funds_rate_1y_ago=body.fed_funds_rate_1y_ago,
        ig_credit_spread=body.ig_credit_spread,
    )
    return {
        "gdp_regime": snap.gdp_regime,
        "inflation_regime": snap.inflation_regime,
        "monetary_regime": snap.monetary_regime,
        "credit_regime": snap.credit_regime,
        "equity_macro_score": snap.equity_macro_score,
        "macro_environment_score": snap.macro_environment_score,
        "favored_axiom_factors": snap.favored_axiom_factors,
        "unfavored_axiom_factors": snap.unfavored_axiom_factors,
        "macro_regime_label": snap.macro_regime_label,
        "investment_implications": snap.investment_implications,
    }
