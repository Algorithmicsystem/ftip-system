"""Phase 21: Portfolio Allocation Engine.

Takes the conviction-ranked screener output and constructs a portfolio
allocation that respects:
  - Per-position Kelly weight cap
  - Per-sector concentration limit
  - Portfolio heat (total invested weight) cap

No DB writes — pure read-over existing axiom_scores_daily + market_symbols.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from api import db

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class AllocationEntry:
    rank: int
    symbol: str
    sector: str
    signal_label: str
    dau: float
    conviction_score: float
    suggested_weight: float
    suggested_weight_pct: str
    size_band: str
    deployability_tier: str
    ic_state: str
    active_constraint: str
    downside_flags: List[str] = field(default_factory=list)


@dataclass
class RejectedEntry:
    symbol: str
    sector: str
    dau: float
    conviction_score: float
    kelly_weight: float
    rejection_reason: str


# ---------------------------------------------------------------------------
# Core allocation function
# ---------------------------------------------------------------------------

def build_portfolio_allocation(
    as_of_date: dt.date,
    *,
    max_position_weight: float = 0.10,
    max_sector_concentration: float = 0.30,
    max_portfolio_heat: float = 1.0,
    min_dau: float = 0.0,
    min_conviction: float = 0.0,
    fractional_kelly: float = 0.5,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return a sector-capped, heat-limited portfolio allocation.

    Pulls from axiom_scores_daily × prosperity_signals_daily × market_symbols.
    Per-symbol Kelly sizing and sector/heat enforcement happen in Python.
    """
    if not db.db_read_enabled():
        return {
            "status": "db_disabled",
            "as_of_date": as_of_date.isoformat(),
            "allocations": [],
            "rejected": [],
            "portfolio_weight_total": 0.0,
            "position_count": 0,
            "sector_breakdown": {},
        }

    from api.axiom.screener import _load_ic_state_bulk, _load_breadth_state_bulk, _safe_float, _engine_score
    from api.axiom.sizer import compute_kelly_size
    from api.jobs.alerts import compute_conviction_score
    from api.axiom.memo import conviction_tier as get_tier

    ic_state = _load_ic_state_bulk(as_of_date)
    breadth_state = _load_breadth_state_bulk(as_of_date)
    hit_rate = _load_hit_rate()

    conditions = ["a.as_of_date = %s"]
    params: List[Any] = [as_of_date]

    if min_dau > 0:
        conditions.append("(a.payload->>'deployable_alpha_utility')::numeric >= %s")
        params.append(min_dau)

    where_clause = " AND ".join(conditions)
    params.append(500)

    rows = db.safe_fetchall(
        f"""
        SELECT
            a.symbol,
            a.payload,
            COALESCE(p.signal, 'HOLD') AS signal_label,
            COALESCE(m.sector, 'Unknown')  AS sector
        FROM axiom_scores_daily a
        LEFT JOIN prosperity_signals_daily p
            ON p.symbol = a.symbol
            AND p.as_of = a.as_of_date
            AND p.lookback = 252
        LEFT JOIN market_symbols m
            ON m.symbol = a.symbol
        WHERE {where_clause}
        ORDER BY (a.payload->>'deployable_alpha_utility')::numeric DESC NULLS LAST
        LIMIT %s
        """,
        tuple(params),
    )

    # ------------------------------------------------------------------
    # Score and size each candidate
    # ------------------------------------------------------------------
    candidates: List[Tuple[float, AllocationEntry, RejectedEntry]] = []

    for row in (rows or []):
        symbol = str(row[0])
        payload_raw = row[1]
        signal_label = str(row[2] or "HOLD")
        sector = str(row[3] or "Unknown")

        if isinstance(payload_raw, str):
            try:
                payload: Dict = json.loads(payload_raw)
            except Exception:
                continue
        else:
            payload = payload_raw or {}

        dau = _safe_float(payload.get("deployable_alpha_utility"), 0.0)
        if dau < min_dau:
            continue

        fragility_score    = _engine_score(payload, "critical_fragility",  50.0)
        liquidity_score    = _engine_score(payload, "liquidity_convexity",  50.0)
        research_score     = _engine_score(payload, "research_integrity",   50.0)
        overall_confidence = _safe_float(payload.get("overall_confidence"), 50.0)
        deployability_tier = str(payload.get("deployability_tier") or "monitor_only")
        regime_label       = str(payload.get("regime_label") or "unknown")

        conviction = compute_conviction_score(
            dau=dau,
            signal_label=signal_label,
            regime_label=regime_label,
            breadth_state=breadth_state,
            ic_state=ic_state,
        )
        if conviction < min_conviction:
            continue

        sizing = compute_kelly_size(
            symbol=symbol,
            as_of_date=as_of_date.isoformat(),
            dau=dau,
            fragility_score=fragility_score,
            liquidity_score=liquidity_score,
            research_score=research_score,
            overall_confidence=overall_confidence,
            deployability_tier=deployability_tier,
            ic_state=ic_state,
            hit_rate=hit_rate,
            fractional_kelly=fractional_kelly,
            max_weight=max_position_weight,
        )

        entry = AllocationEntry(
            rank=0,
            symbol=symbol,
            sector=sector,
            signal_label=signal_label,
            dau=round(dau, 2),
            conviction_score=round(conviction, 2),
            suggested_weight=sizing.suggested_weight,
            suggested_weight_pct=f"{sizing.suggested_weight * 100:.2f}%",
            size_band=sizing.size_band,
            deployability_tier=deployability_tier,
            ic_state=ic_state,
            active_constraint=sizing.active_constraint,
            downside_flags=sizing.downside_flags,
        )
        rejected_proto = RejectedEntry(
            symbol=symbol,
            sector=sector,
            dau=round(dau, 2),
            conviction_score=round(conviction, 2),
            kelly_weight=sizing.suggested_weight,
            rejection_reason="",
        )
        candidates.append((conviction, entry, rejected_proto))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Apply sector caps and portfolio heat
    # ------------------------------------------------------------------
    sector_weight: Dict[str, float] = {}
    portfolio_heat = 0.0
    allocations: List[AllocationEntry] = []
    rejected: List[RejectedEntry] = []

    for i, (conviction, entry, rej) in enumerate(candidates):
        if len(allocations) >= limit:
            rej.rejection_reason = "limit_reached"
            rejected.append(rej)
            continue

        w = entry.suggested_weight
        if w <= 0.0:
            rej.rejection_reason = "zero_weight"
            rejected.append(rej)
            continue

        if portfolio_heat + w > max_portfolio_heat:
            # Try partial fill
            available = max_portfolio_heat - portfolio_heat
            if available < 0.001:
                rej.rejection_reason = "portfolio_heat_cap"
                rejected.append(rej)
                continue
            w = available

        sec = entry.sector or "Unknown"
        sec_used = sector_weight.get(sec, 0.0)
        if sec_used + w > max_sector_concentration:
            available = max_sector_concentration - sec_used
            if available < 0.001:
                rej.rejection_reason = "sector_cap_exceeded"
                rejected.append(rej)
                continue
            w = available

        entry.suggested_weight = round(w, 6)
        entry.suggested_weight_pct = f"{w * 100:.2f}%"
        if w < entry.suggested_weight:
            entry.active_constraint = "sector_cap" if (sector_weight.get(sec, 0) + w >= max_sector_concentration) else "portfolio_heat"

        entry.rank = len(allocations) + 1
        sector_weight[sec] = round(sector_weight.get(sec, 0.0) + w, 6)
        portfolio_heat = round(portfolio_heat + w, 6)
        allocations.append(entry)

    sector_breakdown = {k: round(v, 4) for k, v in sorted(sector_weight.items(), key=lambda x: -x[1])}

    return {
        "status": "ok",
        "as_of_date": as_of_date.isoformat(),
        "ic_state": ic_state,
        "breadth_state": breadth_state,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "portfolio_weight_total": round(portfolio_heat, 4),
        "portfolio_weight_pct": f"{portfolio_heat * 100:.1f}%",
        "position_count": len(allocations),
        "sector_breakdown": sector_breakdown,
        "constraints": {
            "max_position_weight": max_position_weight,
            "max_sector_concentration": max_sector_concentration,
            "max_portfolio_heat": max_portfolio_heat,
            "fractional_kelly": fractional_kelly,
        },
        "allocations": [asdict(a) for a in allocations],
        "rejected": [asdict(r) for r in rejected if r.rejection_reason],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hit_rate() -> Optional[float]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT payload
            FROM axiom_calibration_snapshots
            WHERE snapshot_key LIKE 'ic_daily_v1:%'
            ORDER BY created_at DESC LIMIT 1
            """,
            (),
        )
        if not row:
            return None
        p = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
        return float(
            (p.get("diagnostics") or {})
            .get("overall_outcome_metrics", {})
            .get("hit_rate") or 0.5
        )
    except Exception:
        return None
