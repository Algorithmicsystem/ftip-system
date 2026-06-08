"""Phase 10.3: Morning Intelligence Briefing.

Assembles a structured intelligence package from previous day's AXIOM outputs.
Runs at 7:30am ET every weekday via the scheduler.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from api import db, security
from api.assistant.phase3.common import clamp

router = APIRouter(
    prefix="/jobs/briefing",
    tags=["briefing"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)
logger = logging.getLogger(__name__)

_SRI_CRITICAL = 75.0
_SRI_ELEVATED = 50.0

_BUBBLE_SCPS_THRESHOLD = 70.0
_BUBBLE_BFS_THRESHOLD = 65.0
_EARNINGS_PESS_THRESHOLD = 65.0
_EARNINGS_HORIZON_DAYS = 30


def _sri_label(sri: float) -> str:
    if sri >= _SRI_CRITICAL:
        return "critical"
    if sri >= _SRI_ELEVATED:
        return "elevated"
    return "normal"


@dataclass
class MorningBriefing:
    briefing_date: dt.date
    market_session: str
    regime_context: Dict[str, Any] = field(default_factory=dict)
    top_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    key_risks: List[Dict[str, Any]] = field(default_factory=list)
    sector_rotation: Dict[str, Any] = field(default_factory=dict)
    bubble_watch: List[Dict[str, Any]] = field(default_factory=list)
    earnings_calendar: List[Dict[str, Any]] = field(default_factory=list)
    factor_environment: Dict[str, Any] = field(default_factory=dict)
    ml_model_health: Dict[str, Any] = field(default_factory=dict)
    systemic_risk_index: float = 50.0
    briefing_text: str = ""
    cross_asset_context: Dict[str, Any] = field(default_factory=dict)


def compute_systemic_risk_index(as_of_date: dt.date) -> float:
    """Compute SRI — delegates to full Phase 11 SRI engine when DB is available.

    Returns 50.0 (neutral) if insufficient data.
    """
    if not db.db_read_enabled():
        return 50.0
    try:
        from api.axiom.risk.systemic_risk import compute_sri
        return compute_sri(as_of_date)["sri"]
    except Exception:
        pass

    try:
        rows = db.safe_fetchall(
            """
            SELECT
                payload->>'engine_scores' AS engine_scores_json
            FROM axiom_scores_daily
            WHERE as_of_date = %s
              AND payload IS NOT NULL
            LIMIT 30
            """,
            (as_of_date,),
        )
    except Exception:
        return 50.0

    if not rows:
        return 50.0

    frag_scores: List[float] = []
    scps_scores: List[float] = []
    bfs_scores: List[float] = []

    for row in rows:
        raw = row[0]
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                es = json.loads(raw)
            except Exception:
                continue
        else:
            es = raw or {}

        frag = (es.get("critical_fragility") or {}).get("score")
        if frag is not None:
            frag_scores.append(float(frag))

        comps = (es.get("critical_fragility") or {}).get("components") or {}
        if comps.get("scps_component") is not None:
            scps_scores.append(float(comps["scps_component"]))
        if comps.get("bfs_component") is not None:
            bfs_scores.append(float(comps["bfs_component"]))

    avg_frag = sum(frag_scores) / len(frag_scores) if frag_scores else 50.0
    avg_scps = sum(scps_scores) / len(scps_scores) if scps_scores else 50.0
    avg_bfs = sum(bfs_scores) / len(bfs_scores) if bfs_scores else 50.0

    # IC degradation
    ic_deg = 0.0
    try:
        ic_row = db.safe_fetchone(
            """
            SELECT ic_state
              FROM signal_ic_daily
             WHERE as_of_date <= %s
             ORDER BY as_of_date DESC
             LIMIT 1
            """,
            (as_of_date,),
        )
        if ic_row:
            ic_state = str(ic_row[0] or "INSUFFICIENT").upper()
            ic_deg = {"DEGRADED": 1.0, "WEAK": 0.5}.get(ic_state, 0.0)
    except Exception:
        pass

    # Correlation spike proxy from signal_pnl_daily returns
    corr_spike = 0.5
    try:
        since = as_of_date - dt.timedelta(days=21)
        pnl_rows = db.safe_fetchall(
            """
            SELECT symbol, return_pct
              FROM signal_pnl_daily
             WHERE signal_date >= %s AND signal_date <= %s
               AND return_pct IS NOT NULL AND horizon_days = 5
            """,
            (since, as_of_date),
        )
        if pnl_rows and len(pnl_rows) >= 20:
            rets = [float(r[1]) for r in pnl_rows]
            import statistics
            mean_r = statistics.mean(rets)
            std_r = statistics.stdev(rets) if len(rets) > 1 else 1.0
            # Normalize std to [0,1]: low std = high correlation proxy
            corr_spike = clamp(1.0 - std_r * 10, 0.0, 1.0)
    except Exception:
        pass

    sri_raw = (
        avg_frag * 0.25
        + avg_scps * 0.20
        + avg_bfs * 0.20
        + ic_deg * 100.0 * 0.20
        + corr_spike * 100.0 * 0.15
    )
    return round(clamp(sri_raw, 0.0, 100.0), 2)


def generate_morning_briefing(as_of_date: Optional[dt.date] = None) -> MorningBriefing:
    """Assemble the full morning intelligence briefing."""
    aod = as_of_date or dt.date.today()
    now_hour = dt.datetime.now().hour
    if now_hour < 9:
        session = "pre_market"
    elif now_hour < 16:
        session = "market_open"
    else:
        session = "intraday"

    sri = compute_systemic_risk_index(aod)

    # Default structure for DB-disabled mode
    if not db.db_enabled():
        _dev_regime = (
            "low_risk_moderate_growth" if sri < 40 else
            "moderate_risk_neutral" if sri < 70 else
            "elevated_risk_cautious"
        )
        return MorningBriefing(
            briefing_date=aod,
            market_session=session,
            systemic_risk_index=sri,
            regime_context={"regime_label": _dev_regime},
            briefing_text=_build_text(
                regime=_dev_regime, breadth_state="UNKNOWN", n_favorable=0,
                top_symbol=None, top_dau=0.0, top_driver="—",
                risk_symbol=None, risk_type="—",
                top_factor="—", sri=sri, ic_state="INSUFFICIENT", sample_count=0,
            ),
        )

    # Fetch top opportunities
    top_rows = db.safe_fetchall(
        """
        SELECT symbol,
               (payload->>'deployable_alpha_utility')::numeric AS dau,
               payload->>'regime_label' AS regime,
               payload->>'deployability_tier' AS tier
          FROM axiom_scores_daily
         WHERE as_of_date = %s
           AND (payload->>'deployable_alpha_utility')::numeric > 50
         ORDER BY dau DESC
         LIMIT 5
        """,
        (aod,),
    ) or []

    top_opportunities = [
        {"symbol": r[0], "dau": float(r[1] or 0), "regime": r[2], "tier": r[3]}
        for r in top_rows
    ]

    # Key risks (high fragility)
    risk_rows = db.safe_fetchall(
        """
        SELECT symbol,
               (payload->'engine_scores'->'critical_fragility'->>'score')::numeric AS frag
          FROM axiom_scores_daily
         WHERE as_of_date = %s
           AND (payload->'engine_scores'->'critical_fragility'->>'score')::numeric > 65
         ORDER BY frag DESC
         LIMIT 3
        """,
        (aod,),
    ) or []
    key_risks = [{"symbol": r[0], "fragility_score": float(r[1] or 0)} for r in risk_rows]

    # Bubble watch
    bubble_rows = db.safe_fetchall(
        """
        SELECT symbol,
               (payload->'engine_scores'->'critical_fragility'->'components'->>'scps_component')::numeric AS scps,
               (payload->'engine_scores'->'critical_fragility'->'components'->>'bfs_component')::numeric AS bfs
          FROM axiom_scores_daily
         WHERE as_of_date = %s
           AND (
               (payload->'engine_scores'->'critical_fragility'->'components'->>'scps_component')::numeric > %s
               OR
               (payload->'engine_scores'->'critical_fragility'->'components'->>'bfs_component')::numeric > %s
           )
         LIMIT 10
        """,
        (aod, _BUBBLE_SCPS_THRESHOLD, _BUBBLE_BFS_THRESHOLD),
    ) or []
    bubble_watch = [
        {"symbol": r[0], "scps_score": float(r[1] or 0), "bfs_score": float(r[2] or 0)}
        for r in bubble_rows
    ]

    # IC state
    ic_state = "INSUFFICIENT"
    sample_count = 0
    try:
        ic_row = db.safe_fetchone(
            "SELECT ic_state, sample_size FROM signal_ic_daily WHERE as_of_date <= %s ORDER BY as_of_date DESC LIMIT 1",
            (aod,),
        )
        if ic_row:
            ic_state = str(ic_row[0] or "INSUFFICIENT").upper()
            sample_count = int(ic_row[1] or 0)
    except Exception:
        pass

    # Breadth state
    breadth_state = "UNKNOWN"
    n_favorable = 0
    try:
        b_row = db.safe_fetchone(
            "SELECT breadth_state FROM market_breadth_daily WHERE as_of_date = %s", (aod,)
        )
        if b_row:
            breadth_state = str(b_row[0] or "UNKNOWN").upper()
        n_favorable = len([r for r in top_rows])
    except Exception:
        pass

    # Regime context
    regime_label = top_opportunities[0]["regime"] if top_opportunities else "unknown"
    if not regime_label or regime_label.upper() in ("UNKNOWN", ""):
        if sri < 40:
            regime_label = "low_risk_moderate_growth"
        elif sri < 70:
            regime_label = "moderate_risk_neutral"
        else:
            regime_label = "elevated_risk_cautious"

    # ML model health
    ml_health: Dict[str, Any] = {}
    try:
        from api.axiom.ml.model_registry import get_model_version
        ml_health["model_version"] = get_model_version()
        from api.axiom.ml.drift_monitor import check_model_drift
        drift = check_model_drift(aod)
        ml_health["psi_score"] = drift.get("overall_psi", 0.0)
        ml_health["drift_detected"] = drift.get("drift_detected", False)
        ml_health["recommendation"] = drift.get("recommendation", "stable")
    except Exception:
        ml_health = {"model_version": "unknown", "psi_score": 0.0}

    # Sector rotation
    sector_rows = db.safe_fetchall(
        """
        SELECT COALESCE(m.sector, 'Unknown'), AVG((a.payload->>'deployable_alpha_utility')::numeric)
          FROM axiom_scores_daily a
          LEFT JOIN market_symbols m ON m.symbol = a.symbol
         WHERE a.as_of_date = %s
         GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT 5
        """,
        (aod,),
    ) or []
    sector_rotation = {r[0]: round(float(r[1] or 0), 2) for r in sector_rows}

    # Factor environment
    factor_env: Dict[str, Any] = {}
    try:
        factor_rows = db.safe_fetchall(
            """
            SELECT factor_name, AVG(loading) AS avg_loading
              FROM factor_exposures_daily
             WHERE as_of_date = %s
             GROUP BY 1 ORDER BY avg_loading DESC NULLS LAST LIMIT 3
            """,
            (aod,),
        )
        if factor_rows:
            factor_env["top_factors"] = [
                {"factor": r[0], "avg_loading": round(float(r[1] or 0), 4)} for r in factor_rows
            ]
    except Exception:
        pass

    top_factor = (factor_env.get("top_factors") or [{}])[0].get("factor", "EIF")
    top_symbol = top_opportunities[0]["symbol"] if top_opportunities else None
    top_dau = top_opportunities[0]["dau"] if top_opportunities else 0.0
    risk_symbol = key_risks[0]["symbol"] if key_risks else None
    risk_type = "elevated fragility"

    briefing_text = _build_text(
        regime=regime_label,
        breadth_state=breadth_state,
        n_favorable=n_favorable,
        top_symbol=top_symbol,
        top_dau=top_dau,
        top_driver=top_factor,
        risk_symbol=risk_symbol,
        risk_type=risk_type,
        top_factor=top_factor,
        sri=sri,
        ic_state=ic_state,
        sample_count=sample_count,
    )

    # Cross-asset context
    cross_asset_ctx: Dict[str, Any] = {}
    try:
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        ca = compute_cross_asset_snapshot({}, regime_label)
        cross_asset_ctx = {
            "cross_asset_confirmation_score": ca.cross_asset_confirmation_score,
            "fixed_income_signal": ca.fixed_income_signal,
            "volatility_signal": ca.volatility_signal,
            "equity_signal_amplifier": ca.equity_signal_amplifier,
            "macro_narrative": ca.macro_narrative,
        }
    except Exception:
        pass

    briefing = MorningBriefing(
        briefing_date=aod,
        market_session=session,
        regime_context={"regime_label": regime_label},
        top_opportunities=top_opportunities,
        key_risks=key_risks,
        sector_rotation=sector_rotation,
        bubble_watch=bubble_watch,
        earnings_calendar=[],  # populated when earnings data available
        factor_environment=factor_env,
        ml_model_health=ml_health,
        systemic_risk_index=sri,
        briefing_text=briefing_text,
        cross_asset_context=cross_asset_ctx,
    )

    # Store briefing
    _store_briefing(briefing)
    return briefing


def _build_text(
    regime: str, breadth_state: str, n_favorable: int,
    top_symbol: Optional[str], top_dau: float, top_driver: str,
    risk_symbol: Optional[str], risk_type: str,
    top_factor: str, sri: float, ic_state: str, sample_count: int,
) -> str:
    label = _sri_label(sri)

    # Section A: Market context and regime
    regime_display = (regime or "unknown").replace("_", " ").title()
    breadth_display = (breadth_state or "UNKNOWN").replace("_", " ").lower()
    para1 = (
        f"Market context: The current regime is {regime_display}. "
        f"Breadth is {breadth_display} with {n_favorable} of 30 universe symbols "
        f"showing favorable deployability scores. "
        f"The systemic risk index stands at {sri:.1f} ({label})."
    )

    # Section B: Top signal
    top_str = f"{top_symbol} (DAU {top_dau:.1f})" if top_symbol else "no high-conviction candidate"
    signal_action = "BUY" if top_dau >= 65 else "HOLD"
    para2 = (
        f"Top signal: {top_str} — recommended action {signal_action}, "
        f"primary factor driver is {top_driver}."
    )

    # Section C: Regime intelligence
    para3 = (
        f"Regime intelligence: Factor environment favors {top_factor}. "
        f"IC track record is {ic_state} based on {sample_count} matured signals."
    )

    # Section D: Risk summary
    risk_str = f"{risk_symbol} showing {risk_type}" if risk_symbol else "no critical risk signals detected"
    sri_trend = "elevated" if sri >= _SRI_ELEVATED else "within normal range"
    para4 = (
        f"Risk: {risk_str}. "
        f"Systemic risk is {sri_trend} ({sri:.1f})."
    )

    return f"{para1}\n\n{para2}\n\n{para3}\n\n{para4}"


def _store_briefing(briefing: MorningBriefing) -> None:
    if not db.db_read_enabled():
        return
    try:
        import dataclasses
        payload = dataclasses.asdict(briefing)
        payload["briefing_date"] = briefing.briefing_date.isoformat()
        db.safe_execute(
            """
            INSERT INTO morning_briefings (date, briefing, sri, created_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (date) DO UPDATE
               SET briefing = EXCLUDED.briefing,
                   sri = EXCLUDED.sri,
                   created_at = now()
            """,
            (briefing.briefing_date, json.dumps(payload), briefing.systemic_risk_index),
        )
    except Exception as exc:
        logger.warning("morning_briefing.store_failed error=%s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _load_briefing_from_cache(aod: dt.date) -> Optional[Dict[str, Any]]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            "SELECT briefing FROM morning_briefings WHERE date = %s",
            (aod,),
        )
        if row and row[0]:
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return payload
    except Exception:
        pass
    return None


@router.get("/morning")
def get_morning_briefing(
    as_of_date: Optional[str] = Query(default=None),
    force_refresh: bool = Query(default=False),
) -> Dict[str, Any]:
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    if not force_refresh:
        cached = _load_briefing_from_cache(aod)
        if cached:
            cached["cached"] = True
            return cached
    briefing = generate_morning_briefing(aod)
    import dataclasses
    result = dataclasses.asdict(briefing)
    result["briefing_date"] = str(briefing.briefing_date)
    result["cached"] = False
    return result


@router.post("/morning")
def trigger_morning_briefing(
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    briefing = generate_morning_briefing(aod)
    import dataclasses
    result = dataclasses.asdict(briefing)
    result["briefing_date"] = str(briefing.briefing_date)
    result["cached"] = False
    return result


@router.get("/morning/text")
def get_morning_briefing_text(
    as_of_date: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    aod = dt.date.fromisoformat(as_of_date) if as_of_date else dt.date.today()
    if not (as_of_date and as_of_date != str(dt.date.today())):
        cached = _load_briefing_from_cache(aod)
        if cached:
            return {"briefing_date": str(aod), "text": cached.get("briefing_text", ""), "cached": True}
    briefing = generate_morning_briefing(aod)
    return {"briefing_date": str(briefing.briefing_date), "text": briefing.briefing_text, "cached": False}
