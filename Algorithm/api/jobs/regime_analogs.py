"""Phase 2.1 / 2.3: Regime Analog Library and Symbol Linkage Intelligence.

Provides:
  find_regime_analogs     — closest historical matches by regime + macro context
  compute_macro_sensitivity — quarterly OLS beta of returns on macro factors
  get_peers_with_axiom    — sector peers with current AXIOM scores
  get_stress_propagation  — how stress in a symbol propagates to linked peers
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.1 — Regime Analog Library
# ---------------------------------------------------------------------------

def find_regime_analogs(
    regime_label: str,
    *,
    limit: int = 5,
    vix_current: Optional[float] = None,
    cape_current: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return closest historical regime analogs for a given regime label.

    Similarity scoring (0–100):
    - Regime label match: required filter (not scored)
    - VIX proximity: ±10 points → linearly weighted (30%)
    - CAPE proximity: ±10 points → linearly weighted (30%)
    - Recency bonus: more recent events scored slightly lower (40%)

    Returns ordered by similarity_score DESC, then by reference_date DESC.
    """
    if not db.db_read_enabled():
        return []

    try:
        rows = db.safe_fetchall(
            """
            SELECT analog_id, reference_date, regime_label,
                   macro_context, following_30d_return_by_sector,
                   following_90d_return_by_sector,
                   vix_at_entry, cape_at_entry, ic_state_at_entry, description
            FROM regime_analog_library
            WHERE regime_label = %s
            ORDER BY reference_date DESC
            LIMIT %s
            """,
            (regime_label, limit * 4),  # over-fetch, then re-rank
        )
    except Exception as exc:
        logger.warning("regime_analogs.find_failed error=%s", exc)
        return []

    analogs = []
    for row in rows:
        (analog_id, ref_date, regime, macro_ctx, ret_30d, ret_90d,
         vix, cape, ic_state, description) = row

        # Compute similarity score
        score = 100.0

        if vix_current is not None and vix is not None:
            vix_diff = abs(float(vix_current) - float(vix))
            score -= min(vix_diff / 10.0, 1.0) * 30.0  # up to -30 pts

        if cape_current is not None and cape is not None:
            cape_diff = abs(float(cape_current) - float(cape))
            score -= min(cape_diff / 10.0, 1.0) * 30.0  # up to -30 pts

        # Recency weight: 2000→0 pts, 2025→full pts
        ref_year = ref_date.year if hasattr(ref_date, "year") else 2010
        recency_factor = min(max((ref_year - 2000) / 25.0, 0.0), 1.0)
        score = score * (0.60 + 0.40 * recency_factor)

        analogs.append({
            "analog_id": analog_id,
            "reference_date": ref_date.isoformat() if hasattr(ref_date, "isoformat") else str(ref_date),
            "regime_label": regime,
            "macro_context": macro_ctx or {},
            "following_30d_return_by_sector": ret_30d or {},
            "following_90d_return_by_sector": ret_90d or {},
            "vix_at_entry": float(vix) if vix is not None else None,
            "cape_at_entry": float(cape) if cape is not None else None,
            "ic_state_at_entry": ic_state,
            "description": description,
            "similarity_score": round(score, 1),
        })

    analogs.sort(key=lambda x: -x["similarity_score"])
    return analogs[:limit]


# ---------------------------------------------------------------------------
# 2.2 — Company-Macro Sensitivity
# ---------------------------------------------------------------------------

def load_macro_sensitivity(symbol: str) -> List[Dict[str, Any]]:
    """Load most recent macro sensitivity betas for a symbol."""
    if not db.db_read_enabled():
        return []

    try:
        rows = db.safe_fetchall(
            """
            SELECT macro_factor, sensitivity_beta, r_squared, lookback_days, estimated_at
            FROM company_macro_sensitivity
            WHERE symbol = %s
            ORDER BY estimated_at DESC, macro_factor
            LIMIT 50
            """,
            (symbol,),
        )
    except Exception as exc:
        logger.warning("macro_sensitivity.load_failed symbol=%s error=%s", symbol, exc)
        return []

    return [
        {
            "macro_factor": r[0],
            "sensitivity_beta": float(r[1]) if r[1] is not None else None,
            "r_squared": float(r[2]) if r[2] is not None else None,
            "lookback_days": r[3],
            "estimated_at": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
        }
        for r in rows
    ]


def store_macro_sensitivity(
    symbol: str,
    macro_factor: str,
    sensitivity_beta: float,
    r_squared: Optional[float],
    estimated_at: dt.date,
    lookback_days: int = 252,
) -> bool:
    """Upsert a single macro sensitivity row."""
    if not db.db_write_enabled():
        return False
    try:
        db.safe_execute(
            """
            INSERT INTO company_macro_sensitivity
                (symbol, macro_factor, sensitivity_beta, r_squared, lookback_days, estimated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, macro_factor, estimated_at)
            DO UPDATE SET
                sensitivity_beta = EXCLUDED.sensitivity_beta,
                r_squared        = EXCLUDED.r_squared,
                lookback_days    = EXCLUDED.lookback_days,
                updated_at       = now()
            """,
            (symbol, macro_factor, sensitivity_beta, r_squared, lookback_days, estimated_at),
        )
        return True
    except Exception as exc:
        logger.warning("macro_sensitivity.store_failed symbol=%s error=%s", symbol, exc)
        return False


def compute_ols_beta(returns: List[float], factor: List[float]) -> Dict[str, float]:
    """Compute OLS beta and R² of returns on a macro factor series."""
    n = min(len(returns), len(factor))
    if n < 10:
        return {"beta": 0.0, "r_squared": 0.0}

    r = returns[:n]
    f = factor[:n]

    r_mean = sum(r) / n
    f_mean = sum(f) / n

    cov_rf = sum((r[i] - r_mean) * (f[i] - f_mean) for i in range(n))
    var_f = sum((f[i] - f_mean) ** 2 for i in range(n))

    if var_f <= 0:
        return {"beta": 0.0, "r_squared": 0.0}

    beta = cov_rf / var_f
    fitted = [f_mean + beta * (f[i] - f_mean) for i in range(n)]
    ss_res = sum((r[i] - fitted[i]) ** 2 for i in range(n))
    ss_tot = sum((r[i] - r_mean) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"beta": round(beta, 4), "r_squared": round(max(r2, 0.0), 4)}


# ---------------------------------------------------------------------------
# 2.3 — Cross-Sector Linkage Intelligence
# ---------------------------------------------------------------------------

def get_peers_with_axiom(symbol: str, as_of_date: dt.date) -> List[Dict[str, Any]]:
    """Return active sector peers for a symbol with their current AXIOM scores."""
    if not db.db_read_enabled():
        return []

    try:
        rows = db.safe_fetchall(
            """
            SELECT
                sl.linked_symbol,
                sl.link_type,
                sl.linkage_strength,
                sl.last_validated,
                sl.validation_method,
                asd.deployable_alpha_utility,
                asd.overall_confidence,
                asd.deployability_tier,
                asd.regime_label
            FROM symbol_linkage sl
            LEFT JOIN axiom_scores_daily asd
                ON  asd.symbol     = sl.linked_symbol
                AND asd.as_of_date = (
                    SELECT MAX(as_of_date) FROM axiom_scores_daily
                    WHERE symbol = sl.linked_symbol AND as_of_date <= %s
                )
            WHERE sl.symbol = %s AND sl.is_active = TRUE
            ORDER BY sl.link_type, sl.linkage_strength DESC NULLS LAST
            """,
            (as_of_date, symbol),
        )
    except Exception as exc:
        logger.warning("linkage.peers_failed symbol=%s error=%s", symbol, exc)
        return []

    return [
        {
            "linked_symbol": r[0],
            "link_type": r[1],
            "linkage_strength": float(r[2]) if r[2] is not None else None,
            "last_validated": r[3].isoformat() if r[3] and hasattr(r[3], "isoformat") else None,
            "validation_method": r[4],
            "dau": float(r[5]) if r[5] is not None else None,
            "confidence": float(r[6]) if r[6] is not None else None,
            "deployability_tier": r[7],
            "regime_label": r[8],
        }
        for r in rows
    ]


def get_stress_propagation(symbol: str, as_of_date: dt.date) -> Dict[str, Any]:
    """Compute how fragility stress in a symbol propagates to linked peers.

    For each active link, multiplies the source symbol's fragility score by
    the linkage_strength to estimate stress transmission.
    """
    if not db.db_read_enabled():
        return {"status": "db_disabled", "symbol": symbol, "propagation": []}

    # Source fragility
    try:
        src_row = db.safe_fetchone(
            """
            SELECT payload->>'critical_fragility' AS frag
            FROM axiom_scores_daily
            WHERE symbol = %s AND as_of_date <= %s
            ORDER BY as_of_date DESC
            LIMIT 1
            """,
            (symbol, as_of_date),
        )
    except Exception:
        src_row = None

    source_fragility: Optional[float] = None
    if src_row and src_row[0] is not None:
        try:
            import json
            frag_data = json.loads(src_row[0]) if isinstance(src_row[0], str) else src_row[0]
            source_fragility = float(frag_data.get("score", 50.0))
        except (TypeError, ValueError, AttributeError):
            source_fragility = None

    # Peers and their AXIOM scores
    peers = get_peers_with_axiom(symbol, as_of_date)

    propagation = []
    for peer in peers:
        strength = peer.get("linkage_strength") or 0.5
        # Transmitted stress = source_fragility × linkage_strength
        transmitted = (
            round(source_fragility * strength, 1)
            if source_fragility is not None
            else None
        )
        alert = (
            transmitted is not None
            and transmitted > 50.0
            and peer.get("dau") is not None
            and peer["dau"] > 60.0
        )
        propagation.append({
            **peer,
            "transmitted_stress": transmitted,
            "stress_alert": alert,
        })

    propagation.sort(key=lambda x: -(x.get("transmitted_stress") or 0.0))

    return {
        "status": "ok",
        "symbol": symbol,
        "as_of_date": as_of_date.isoformat(),
        "source_fragility": source_fragility,
        "propagation": propagation,
    }
