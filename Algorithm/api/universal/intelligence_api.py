"""Phase 17.1: Universal Intelligence Endpoint — single pre-computed assembly."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from api import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache with TTL (avoids DB round-trip on hot path)
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS = 300  # 5-minute TTL

# {symbol: (timestamp_float, UniversalIntelligenceResponse)}
_cache: Dict[str, Tuple[float, Any]] = {}


def _get_from_memory_cache(symbol: str) -> Optional[Any]:
    """Return cached response if still within TTL, else evict and return None."""
    entry = _cache.get(symbol)
    if entry is None:
        return None
    ts, response = entry
    if time.time() - ts < CACHE_TTL_SECONDS:
        return response
    del _cache[symbol]
    return None


def _set_memory_cache(symbol: str, response: Any) -> None:
    _cache[symbol] = (time.time(), response)


def get_cache_stats() -> Dict[str, Any]:
    now = time.time()
    live = [(sym, ts, resp) for sym, (ts, resp) in _cache.items()
            if now - ts < CACHE_TTL_SECONDS]
    if not live:
        return {
            "cache_type": "in_memory",
            "cache_size": 0,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "oldest_entry_age_seconds": None,
            "newest_entry_age_seconds": None,
            "hit_rate_estimate": "unknown",
        }
    ages = [now - ts for _, ts, _ in live]
    return {
        "cache_type": "in_memory",
        "cache_size": len(live),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "oldest_entry_age_seconds": round(max(ages), 1),
        "newest_entry_age_seconds": round(min(ages), 1),
        "hit_rate_estimate": "unknown",
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UniversalIntelligenceResponse:
    symbol: str
    as_of_date: dt.date

    # Core signal
    signal_label: str
    dau: float
    ml_adjusted_dau: float
    analyst_rating: str
    conviction: str

    # Regime context
    regime_label: str
    regime_strength: float
    systemic_risk_index: float

    # Intelligence quality
    ic_state: str
    intelligence_quality_score: float
    days_of_live_data: int

    # Key scores
    eis_score: float
    caps_score: float
    fragility_score: float
    scps_score: float
    bfs_score: float
    factor_composite_score: float

    # Alternative data
    osms_score: Optional[float]
    ias_score: Optional[float]
    pess_score: Optional[float]

    # Risk
    var_1d_99: Optional[float]
    sri: Optional[float]

    # Explanation
    primary_driver: str
    primary_conclusion: str
    top_supporting_evidence: List[Any]
    top_risk: str

    # Intelligence memory
    signal_batting_average: Optional[float]
    dossier_event_count: int
    moat_score: Optional[float]

    # Staleness
    data_freshness_hours: float
    staleness_warning: bool

    # Cross-asset overlay
    cross_asset_amplifier: float = 0.0
    cross_asset_adjusted_dau: Optional[float] = None
    macro_narrative: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyst_rating(dau: float, signal_label: str) -> str:
    if signal_label == "HOLD":
        return "Hold"
    if signal_label == "BUY":
        if dau > 80:
            return "Strong Buy"
        if dau >= 65:
            return "Buy"
        return "Hold"
    if signal_label == "SELL":
        if dau < 35:
            return "Strong Sell"
        if dau <= 50:
            return "Sell"
        return "Hold"
    return "Hold"


def _signal_from_dau(dau: float) -> str:
    if dau >= 65:
        return "BUY"
    if dau <= 40:
        return "SELL"
    return "HOLD"


def _conviction_from_ic(ic_state: str) -> str:
    if ic_state in ("STRONG", "MODERATE"):
        return "High"
    if ic_state == "WEAK":
        return "Moderate"
    return "Low"


def _compute_freshness_hours(as_of_date: dt.date) -> float:
    delta = dt.date.today() - as_of_date
    return float(delta.days * 24)


def _extract_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    engines = payload.get("engine_scores") or {}
    fundamental = engines.get("fundamental_reality") or {}
    fund_comps = fundamental.get("components") or {}
    fragility = engines.get("critical_fragility") or {}
    frag_comps = fragility.get("components") or {}
    behavioral = engines.get("behavioral_distortion") or {}
    beh_comps = behavioral.get("components") or {}
    liq = engines.get("liquidity_convexity") or {}
    liq_comps = liq.get("components") or {}
    alpha = payload.get("alpha_decomposition") or {}
    factor_contributions = alpha.get("factor_contributions") or {}

    # Top supporting evidence (factors with highest positive contribution)
    pos = [(k, v) for k, v in factor_contributions.items() if v > 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    top_supporting = [{"factor": k, "contribution": round(v, 2)} for k, v in pos[:3]]

    # Top risk (factor with most negative contribution)
    neg = [(k, v) for k, v in factor_contributions.items() if v < 0]
    top_risk = min(neg, key=lambda x: x[1])[0] if neg else "none"

    # Primary driver: prefer stored value; fall back to engine with highest deviation from neutral
    stored_driver = str(alpha.get("primary_driver") or "").strip()
    if stored_driver and stored_driver != "unknown":
        primary_driver = stored_driver
    elif engines:
        best = max(
            ((name, abs(float((data.get("score") or 50.0)) - 50.0))
             for name, data in engines.items() if isinstance(data, dict)),
            key=lambda t: t[1],
            default=(None, 0.0),
        )
        primary_driver = best[0] or "unknown"
    else:
        primary_driver = "unknown"

    return {
        "eis_score": float(fund_comps.get("eis_component") or 50.0),
        "caps_score": float(fund_comps.get("caps_component") or 50.0),
        "fragility_score": float(fragility.get("score") or 50.0),
        "scps_score": float(frag_comps.get("scps_component") or 50.0),
        "bfs_score": float(frag_comps.get("bfs_component") or 50.0),
        "osms_score": float(liq_comps["osms_component"]) if liq_comps.get("osms_component") is not None else None,
        "ias_score": float(liq_comps["ias_component"]) if liq_comps.get("ias_component") is not None else None,
        "pess_score": float(beh_comps.get("pess_component")) if beh_comps.get("pess_component") is not None else None,
        "primary_driver": primary_driver,
        "top_supporting_evidence": top_supporting,
        "top_risk": top_risk,
    }


def _default_response(symbol: str, as_of_date: dt.date) -> UniversalIntelligenceResponse:
    freshness = _compute_freshness_hours(as_of_date)
    return UniversalIntelligenceResponse(
        symbol=symbol,
        as_of_date=as_of_date,
        signal_label="HOLD",
        dau=50.0,
        ml_adjusted_dau=50.0,
        analyst_rating="Hold",
        conviction="Low",
        regime_label="UNKNOWN",
        regime_strength=0.5,
        systemic_risk_index=50.0,
        ic_state="INSUFFICIENT",
        intelligence_quality_score=0.0,
        days_of_live_data=0,
        eis_score=50.0,
        caps_score=50.0,
        fragility_score=50.0,
        scps_score=50.0,
        bfs_score=50.0,
        factor_composite_score=50.0,
        osms_score=None,
        ias_score=None,
        pess_score=None,
        var_1d_99=None,
        sri=None,
        primary_driver="unknown",
        primary_conclusion=f"{symbol} — insufficient data for signal generation.",
        top_supporting_evidence=[],
        top_risk="insufficient_data",
        signal_batting_average=None,
        dossier_event_count=0,
        moat_score=None,
        data_freshness_hours=freshness,
        staleness_warning=freshness > 96.0,
    )


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def assemble_universal_intelligence(
    symbol: str,
    as_of_date: Optional[dt.date] = None,
) -> UniversalIntelligenceResponse:
    if as_of_date is None:
        # Use the most recent scored date, not today — prevents weekend/holiday misses
        if db.db_enabled():
            try:
                row = db.safe_fetchone("SELECT MAX(as_of_date) FROM axiom_scores_daily")
                as_of_date = row[0] if row and row[0] else dt.date.today()
            except Exception:
                as_of_date = dt.date.today()
        else:
            as_of_date = dt.date.today()

    if not db.db_enabled():
        return _default_response(symbol, as_of_date)

    try:
        # Step 1: Latest AXIOM score
        axiom_row = db.safe_fetchone(
            """
            SELECT payload, as_of_date,
                   payload->>'deployable_alpha_utility' AS dau,
                   payload->>'ic_state' AS ic_state,
                   payload->>'regime_label' AS regime,
                   (payload->>'regime_strength')::numeric AS regime_strength,
                   (payload->>'amqs_score')::numeric AS amqs
              FROM axiom_scores_daily
             WHERE symbol = %s
             ORDER BY as_of_date DESC
             LIMIT 1
            """,
            (symbol,),
        )

        if not axiom_row or not axiom_row[0]:
            return _default_response(symbol, as_of_date)

        raw = axiom_row[0]
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
        else:
            payload = {}
        score_date = axiom_row[1] or as_of_date
        dau = float(axiom_row[2] or 50.0)
        ic_state = str(axiom_row[3] or "INSUFFICIENT")
        regime_label = str(axiom_row[4] or "UNKNOWN")
        regime_strength = float(axiom_row[5] or 0.5)
        signal_label = _signal_from_dau(dau)

        # Step 2: ML-adjusted DAU
        ml_row = db.safe_fetchone(
            """
            SELECT ml_adjusted_dau
              FROM axiom_scores_daily
             WHERE symbol = %s
             ORDER BY as_of_date DESC LIMIT 1
            """,
            (symbol,),
        )
        ml_adjusted_dau = float(ml_row[0]) if ml_row and ml_row[0] is not None else dau

        # Step 3: IC state (already from AXIOM row)

        # Step 4: SRI
        sri_row = db.safe_fetchone(
            "SELECT sri FROM market_breadth_daily ORDER BY as_of_date DESC LIMIT 1",
        )
        sri_val = float(sri_row[0]) if sri_row and sri_row[0] is not None else None
        systemic_risk_index = sri_val or 50.0

        # Step 5: VaR (portfolio_risk_daily uses portfolio_id, symbol may not match)
        var_1d_99 = None
        try:
            var_row = db.safe_fetchone(
                """
                SELECT var_99_1d
                  FROM portfolio_risk_daily
                 WHERE portfolio_id = %s
                 ORDER BY as_of_date DESC LIMIT 1
                """,
                (symbol,),
            )
            var_1d_99 = float(var_row[0]) if var_row and var_row[0] is not None else None
        except Exception:
            pass

        # Step 6 + 7: From payload
        extracted = _extract_from_payload(payload)

        # FCS from payload
        factor_composite = float((payload.get("alpha_decomposition") or {}).get("systematic_contribution") or 50.0)

        # Step 8: Batting average
        bat_row = db.safe_fetchone(
            """
            SELECT AVG(batting_average)
              FROM signal_performance_archive
             WHERE symbol = %s AND batting_average IS NOT NULL
            """,
            (symbol,),
        )
        batting = float(bat_row[0]) if bat_row and bat_row[0] is not None else None

        # Step 9-10: Dossier
        dossier_row = db.safe_fetchone(
            """
            SELECT COUNT(*), MAX(impact_score)
              FROM company_intelligence_archive
             WHERE symbol = %s
            """,
            (symbol,),
        )
        dossier_count = int(dossier_row[0]) if dossier_row and dossier_row[0] else 0
        intel_quality = float(dossier_row[1]) if dossier_row and dossier_row[1] is not None else 0.0

        # Step 11: Freshness
        freshness = _compute_freshness_hours(score_date if isinstance(score_date, dt.date) else as_of_date)

        # Step 12: Cross-asset overlay
        cross_amplifier = 0.0
        cross_adjusted_dau = None
        macro_narrative_txt = None
        try:
            from api.macro.cross_asset_engine import compute_cross_asset_snapshot
            from api.assistant.phase3.common import clamp as _clamp
            ca = compute_cross_asset_snapshot({}, regime_label)
            cross_amplifier = ca.equity_signal_amplifier
            cross_adjusted_dau = round(_clamp(dau * (1.0 + cross_amplifier), 0.0, 100.0), 2)
            macro_narrative_txt = ca.macro_narrative
        except Exception:
            pass

        # Primary conclusion from reasoning
        primary_conclusion = (
            f"{symbol} receives a {signal_label} signal with DAU {dau:.1f}; "
            f"primary driver is {extracted['primary_driver']} in a {regime_label} regime."
        )

        return UniversalIntelligenceResponse(
            symbol=symbol,
            as_of_date=score_date if isinstance(score_date, dt.date) else as_of_date,
            signal_label=signal_label,
            dau=dau,
            ml_adjusted_dau=ml_adjusted_dau,
            analyst_rating=_analyst_rating(dau, signal_label),
            conviction=_conviction_from_ic(ic_state),
            regime_label=regime_label,
            regime_strength=regime_strength,
            systemic_risk_index=systemic_risk_index,
            ic_state=ic_state,
            intelligence_quality_score=intel_quality,
            days_of_live_data=0,
            eis_score=extracted["eis_score"],
            caps_score=extracted["caps_score"],
            fragility_score=extracted["fragility_score"],
            scps_score=extracted["scps_score"],
            bfs_score=extracted["bfs_score"],
            factor_composite_score=factor_composite,
            osms_score=extracted["osms_score"],
            ias_score=extracted["ias_score"],
            pess_score=extracted["pess_score"],
            var_1d_99=var_1d_99,
            sri=sri_val,
            primary_driver=extracted["primary_driver"],
            primary_conclusion=primary_conclusion,
            top_supporting_evidence=extracted["top_supporting_evidence"],
            top_risk=extracted["top_risk"],
            signal_batting_average=batting,
            dossier_event_count=dossier_count,
            moat_score=None,
            data_freshness_hours=freshness,
            staleness_warning=freshness > 96.0,
            cross_asset_amplifier=cross_amplifier,
            cross_asset_adjusted_dau=cross_adjusted_dau,
            macro_narrative=macro_narrative_txt,
        )

    except Exception as exc:
        logger.warning("assemble_universal_intelligence_failed symbol=%s err=%s", symbol, exc)
        return _default_response(symbol, as_of_date)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_universal_response(symbol: str, response: UniversalIntelligenceResponse) -> bool:
    if not db.db_read_enabled():
        return False
    try:
        payload = {
            "symbol": response.symbol,
            "as_of_date": response.as_of_date.isoformat(),
            "signal_label": response.signal_label,
            "dau": response.dau,
            "ml_adjusted_dau": response.ml_adjusted_dau,
            "analyst_rating": response.analyst_rating,
            "conviction": response.conviction,
            "regime_label": response.regime_label,
            "ic_state": response.ic_state,
            "data_freshness_hours": response.data_freshness_hours,
            "staleness_warning": response.staleness_warning,
        }
        db.safe_execute(
            """
            INSERT INTO universal_intelligence_cache (symbol, as_of_date, response, assembled_at)
            VALUES (%s, %s, %s::jsonb, now())
            ON CONFLICT (symbol, as_of_date) DO UPDATE
               SET response = EXCLUDED.response,
                   assembled_at = now()
            """,
            (symbol, response.as_of_date, json.dumps(payload)),
        )
        return True
    except Exception as exc:
        logger.debug("cache_universal_response_failed symbol=%s err=%s", symbol, exc)
        return False


def get_cached_universal_response(
    symbol: str,
    as_of_date: dt.date,
) -> Optional[Dict[str, Any]]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT response
              FROM universal_intelligence_cache
             WHERE symbol = %s
               AND as_of_date = %s
               AND assembled_at >= now() - interval '30 minutes'
            """,
            (symbol, as_of_date),
        )
        if row and row[0]:
            return dict(row[0]) if isinstance(row[0], dict) else None
        return None
    except Exception:
        return None
