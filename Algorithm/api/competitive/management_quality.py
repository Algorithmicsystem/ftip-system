"""Phase 18.2: Management Quality Score (MQS)."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ManagementQuality:
    symbol: str
    as_of_date: dt.date

    mqs_score: float

    capital_allocation_score: float
    guidance_accuracy_score: float
    insider_alignment_score: float
    compensation_alignment_score: float
    communication_quality_score: float

    mqs_trend: str
    management_integrity_signal: str

    management_red_flags: List[str]
    management_green_flags: List[str]


# ---------------------------------------------------------------------------
# Component functions (pure — exposed for testing)
# ---------------------------------------------------------------------------

def compute_capital_allocation_score(financials_history: List[Dict]) -> float:
    if not financials_history or len(financials_history) < 2:
        return 50.0

    roics = [float(f.get("roic", 0.0)) for f in financials_history]
    most_recent_roic = roics[-1]
    prior_avg = sum(roics[:-1]) / len(roics[:-1])

    roic_trend = (most_recent_roic - prior_avg) / max(abs(prior_avg), 0.01)
    buyback_timing_score = 50.0  # neutral default

    score = clamp(
        50.0 + roic_trend * 100.0 * 0.60 + (buyback_timing_score - 50.0) * 0.40,
        0.0, 100.0,
    )
    return round(score, 2)


def compute_guidance_accuracy_score(earnings_history: Optional[List[Dict]]) -> float:
    if not earnings_history:
        return 50.0

    within_5pct = 0
    for e in earnings_history:
        guided = float(e.get("guided_eps", 0.0))
        actual = float(e.get("actual_eps", 0.0))
        miss = abs(guided - actual) / max(abs(actual), 0.01)
        if miss <= 0.05:
            within_5pct += 1

    return round(100.0 * within_5pct / len(earnings_history), 2)


def compute_insider_alignment_score(insider_data: Optional[Dict]) -> float:
    if not insider_data:
        return 50.0

    buy_count = int(insider_data.get("buy_count", 0))
    sell_count = int(insider_data.get("sell_count", 0))
    net = buy_count - sell_count

    if net > 3:
        return 90.0
    if net > 0:
        return 65.0
    if net == 0:
        return 50.0
    if net > -3:
        return 35.0
    return 15.0


def _mqs_integrity_signal(mqs: float) -> str:
    if mqs > 65:
        return "high"
    if mqs >= 40:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def compute_mqs(
    symbol: str,
    axiom_payload: Dict[str, Any],
    financials_history: Optional[List[Dict]] = None,
    insider_data: Optional[Dict] = None,
    earnings_history: Optional[List[Dict]] = None,
    as_of_date: Optional[dt.date] = None,
) -> ManagementQuality:
    as_of_date = as_of_date or dt.date.today()

    capital_alloc = compute_capital_allocation_score(financials_history or [])
    guidance_acc = compute_guidance_accuracy_score(earnings_history)
    insider_align = compute_insider_alignment_score(insider_data)

    # Compensation alignment: proxy from Schilit Category 5 (cash quality)
    # If EIS is high, management probably isn't padding earnings → good compensation alignment
    engines = axiom_payload.get("engine_scores") or {}
    fundamental = engines.get("fundamental_reality") or {}
    fund_comps = fundamental.get("components") or {}
    eis = float(fund_comps.get("eis_component") or 50.0)
    comp_align = clamp(eis * 0.6 + 20.0, 0.0, 100.0)

    # Communication quality: proxy from narrative intelligence consistency
    # Use behavioral engine score as proxy
    behavioral = engines.get("behavioral_distortion") or {}
    beh_score = float(behavioral.get("score") or 50.0)
    comm_quality = clamp(beh_score * 0.5 + 25.0, 0.0, 100.0)

    mqs = (
        capital_alloc * 0.35
        + guidance_acc * 0.25
        + insider_align * 0.20
        + comp_align * 0.10
        + comm_quality * 0.10
    )
    mqs = round(clamp(mqs, 0.0, 100.0), 2)

    integrity = _mqs_integrity_signal(mqs)

    red_flags: List[str] = []
    green_flags: List[str] = []

    if capital_alloc < 40:
        red_flags.append("declining_roic_trend")
    elif capital_alloc > 70:
        green_flags.append("improving_roic_trend")

    if insider_align < 35:
        red_flags.append("insider_selling_pattern")
    elif insider_align > 70:
        green_flags.append("insider_buying_signal")

    if guidance_acc < 50:
        red_flags.append("guidance_miss_pattern")
    elif guidance_acc == 100:
        green_flags.append("perfect_guidance_accuracy")

    return ManagementQuality(
        symbol=symbol,
        as_of_date=as_of_date,
        mqs_score=mqs,
        capital_allocation_score=capital_alloc,
        guidance_accuracy_score=guidance_acc,
        insider_alignment_score=insider_align,
        compensation_alignment_score=round(comp_align, 2),
        communication_quality_score=round(comm_quality, 2),
        mqs_trend="stable",
        management_integrity_signal=integrity,
        management_red_flags=red_flags,
        management_green_flags=green_flags,
    )


def get_sector_mqs_rankings(sector: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT symbol, payload
              FROM axiom_scores_daily
             WHERE payload->>'sector' = %s
             ORDER BY as_of_date DESC LIMIT %s
            """,
            (sector, limit),
        ) or []
        rankings = []
        for r in rows:
            sym = str(r[0])
            payload = r[1] if isinstance(r[1], dict) else {}
            mqs = compute_mqs(sym, payload)
            rankings.append({"symbol": sym, "mqs_score": mqs.mqs_score, "integrity": mqs.management_integrity_signal})
        rankings.sort(key=lambda x: x["mqs_score"], reverse=True)
        return rankings
    except Exception as exc:
        logger.warning("sector_mqs_rankings_failed sector=%s err=%s", sector, exc)
        return []
