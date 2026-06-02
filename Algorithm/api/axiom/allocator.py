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
    correlation_threshold: float = 0.80,
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
    from ftip.risk import compute_return_correlation_matrix, correlation_guard

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

    # ------------------------------------------------------------------
    # Correlation guard pass
    # ------------------------------------------------------------------
    corr_adjusted_count = 0
    corr_dropped: List[str] = []

    if len(allocations) >= 2:
        syms = [a.symbol for a in allocations]
        returns_map = _load_returns_for_symbols(syms, as_of_date)
        if len(returns_map) >= 2:
            corr_matrix = compute_return_correlation_matrix(returns_map)
            raw_weights = {a.symbol: a.suggested_weight for a in allocations}
            adjusted = correlation_guard(raw_weights, corr_matrix, threshold=correlation_threshold)
            total_heat_before = sum(raw_weights.values())
            # Scale normalised output back to original heat envelope
            for entry in allocations[:]:
                norm_w = adjusted.get(entry.symbol)
                if norm_w is None or norm_w * total_heat_before < 1e-5:
                    corr_dropped.append(entry.symbol)
                    rej = RejectedEntry(
                        symbol=entry.symbol, sector=entry.sector,
                        dau=entry.dau, conviction_score=entry.conviction_score,
                        kelly_weight=entry.suggested_weight,
                        rejection_reason="correlation_guard",
                    )
                    rejected.append(rej)
                    allocations.remove(entry)
                else:
                    new_w = round(norm_w * total_heat_before, 6)
                    if abs(new_w - entry.suggested_weight) > 1e-6:
                        entry.suggested_weight = new_w
                        entry.suggested_weight_pct = f"{new_w * 100:.2f}%"
                        entry.active_constraint = "correlation_guard"
                        corr_adjusted_count += 1

            # Re-rank and recompute heat/sector after adjustment
            portfolio_heat = 0.0
            sector_weight = {}
            for i, entry in enumerate(allocations):
                entry.rank = i + 1
                w = entry.suggested_weight
                portfolio_heat = round(portfolio_heat + w, 6)
                sec = entry.sector or "Unknown"
                sector_weight[sec] = round(sector_weight.get(sec, 0.0) + w, 6)

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
            "correlation_threshold": correlation_threshold,
        },
        "correlation_guard": {
            "adjusted_count": corr_adjusted_count,
            "dropped_symbols": corr_dropped,
        },
        "allocations": [asdict(a) for a in allocations],
        "rejected": [asdict(r) for r in rejected if r.rejection_reason],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_returns_for_symbols(
    symbols: List[str],
    as_of_date: dt.date,
    lookback_days: int = 60,
) -> Dict[str, List[float]]:
    """Return {symbol: [return_pct, ...]} for pairwise correlation computation.

    Sector-proxy fallback: symbols with < 5 own observations get the average
    return series of other symbols in the same sector that do have enough data.
    This prevents the correlation guard from being skipped entirely when live
    PnL history is thin.
    """
    if not db.db_read_enabled() or not symbols:
        return {}
    since = as_of_date - dt.timedelta(days=lookback_days)
    rows = db.safe_fetchall(
        """
        SELECT p.symbol, p.return_pct, COALESCE(m.sector, 'Unknown') AS sector
        FROM signal_pnl_daily p
        LEFT JOIN market_symbols m ON m.symbol = p.symbol
        WHERE p.symbol = ANY(%s)
          AND p.signal_date >= %s
          AND p.return_pct IS NOT NULL
          AND p.horizon_days = 5
        ORDER BY p.symbol, p.signal_date
        """,
        (symbols, since),
    )
    symbol_sector: Dict[str, str] = {}
    raw: Dict[str, List[float]] = {}
    for row in (rows or []):
        sym = str(row[0])
        ret = row[1]
        sector = str(row[2]) if len(row) > 2 and row[2] is not None else "Unknown"
        symbol_sector[sym] = sector
        raw.setdefault(sym, []).append(float(ret))

    # Symbols with enough data are used directly
    result: Dict[str, List[float]] = {s: v for s, v in raw.items() if len(v) >= 5}

    # Sector proxy fallback for symbols with < 5 data points
    if len(result) >= 2:
        sector_series: Dict[str, List[float]] = {}
        for sym, series in result.items():
            sec = symbol_sector.get(sym, "Unknown")
            existing = sector_series.get(sec, [])
            # Average element-wise across sector peers (truncate to shortest series)
            if not existing:
                sector_series[sec] = list(series)
            else:
                n = min(len(existing), len(series))
                sector_series[sec] = [
                    (existing[i] + series[i]) / 2.0 for i in range(n)
                ]

        for sym in symbols:
            if sym not in result:
                sec = symbol_sector.get(sym, "Unknown")
                proxy = sector_series.get(sec) or sector_series.get("Unknown")
                if proxy and len(proxy) >= 5:
                    result[sym] = proxy

    return result


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
