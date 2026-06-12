"""
AXIOM Tiered Scoring Engine

Scores symbols at different depths based on their tier:

  Tier 1 (≥$10B market cap): Full 7-engine DAU score, every day
  Tier 2 (≥$2B): EIS + CAPS + RFS only (3 engines), every day
                 Full 7-engine score on Mondays or on-demand
  Tier 3 (rest): RFS only (macro regime), every day
                 Full 7-engine score weekly on Sundays or on-demand

This makes scoring 10,000 symbols feasible within a daily pipeline window.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from api import db
from api.universe_registry import get_symbols_by_tier

logger = logging.getLogger(__name__)

SCORING_WORKERS = 10

TIER_1_ENGINES = None                    # None = all engines (full DAU)
TIER_2_ENGINES = ["eis", "caps", "rfs"]  # earnings quality + capital + macro
TIER_3_ENGINES = ["rfs"]                 # macro regime only


def score_universe_tiered(
    as_of_date: dt.date,
    force_full: bool = False,
) -> Dict[str, Any]:
    """
    Score the full universe using tiered depth.

    Returns: {
        tier1_scored: int,
        tier2_scored: int,
        tier3_scored: int,
        total_scored: int,
        errors: list,
        duration_seconds: float,
    }
    """
    start = time.time()

    tier1_symbols = get_symbols_by_tier(1)
    tier2_symbols = get_symbols_by_tier(2)
    tier3_symbols = get_symbols_by_tier(3)

    is_monday = as_of_date.weekday() == 0
    is_sunday = as_of_date.weekday() == 6

    tier2_full = force_full or is_monday
    tier3_full = force_full or is_sunday

    logger.info(
        "tiered_scorer.start tier1=%d tier2=%d tier3=%d "
        "tier2_full=%s tier3_full=%s date=%s",
        len(tier1_symbols), len(tier2_symbols), len(tier3_symbols),
        tier2_full, tier3_full, as_of_date.isoformat(),
    )

    results: Dict[str, Any] = {
        "tier1_scored": 0,
        "tier2_scored": 0,
        "tier3_scored": 0,
        "total_scored": 0,
        "errors": [],
        "duration_seconds": 0.0,
    }

    t1_ok, t1_errors = _score_symbols_parallel(tier1_symbols, as_of_date, TIER_1_ENGINES, 1)
    results["tier1_scored"] = t1_ok
    results["errors"].extend(t1_errors)

    engines_t2 = TIER_1_ENGINES if tier2_full else TIER_2_ENGINES
    t2_ok, t2_errors = _score_symbols_parallel(tier2_symbols, as_of_date, engines_t2, 2)
    results["tier2_scored"] = t2_ok
    results["errors"].extend(t2_errors)

    engines_t3 = TIER_1_ENGINES if tier3_full else TIER_3_ENGINES
    t3_ok, t3_errors = _score_symbols_parallel(tier3_symbols, as_of_date, engines_t3, 3)
    results["tier3_scored"] = t3_ok
    results["errors"].extend(t3_errors)

    results["total_scored"] = t1_ok + t2_ok + t3_ok
    results["duration_seconds"] = round(time.time() - start, 2)

    logger.info(
        "tiered_scorer.complete total=%d t1=%d t2=%d t3=%d "
        "errors=%d duration=%.1fs",
        results["total_scored"],
        t1_ok, t2_ok, t3_ok,
        len(results["errors"]),
        results["duration_seconds"],
    )
    return results


def _score_symbols_parallel(
    symbols: List[str],
    as_of_date: dt.date,
    engines: Optional[List[str]],
    tier: int,
) -> tuple:
    """Score a list of symbols in parallel. Returns (ok_count, error_list)."""
    if not symbols:
        return 0, []

    ok = 0
    errors: List[Dict[str, Any]] = []
    n_workers = min(SCORING_WORKERS, len(symbols))

    def _score_one(symbol: str) -> bool:
        try:
            return _score_single_symbol(symbol, as_of_date, engines, tier)
        except Exception as exc:
            logger.debug(
                "tiered_scorer.symbol_failed sym=%s tier=%d err=%s", symbol, tier, exc
            )
            return False

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_score_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                if future.result():
                    ok += 1
                else:
                    errors.append({"symbol": sym, "tier": tier})
            except Exception as exc:
                errors.append({"symbol": sym, "tier": tier, "error": str(exc)})

    return ok, errors


def _score_single_symbol(
    symbol: str,
    as_of_date: dt.date,
    engines: Optional[List[str]],
    tier: int,
) -> bool:
    """
    Score a single symbol and write to axiom_scores_daily.

    engines=None → full 7-engine score via run_axiom_replay.
    engines=[...] → partial score via _score_partial.
    """
    if engines is None:
        from api.axiom.replay import run_axiom_replay
        run_axiom_replay(
            symbols=[symbol],
            start_date=as_of_date.isoformat(),
            end_date=as_of_date.isoformat(),
            lookback=252,
            persist=True,
        )
        return True
    else:
        return _score_partial(symbol, as_of_date, engines, tier)


def _score_partial(
    symbol: str,
    as_of_date: dt.date,
    engines: List[str],
    tier: int,
) -> bool:
    """
    Compute a partial DAU score using only the specified engines.

    Writes a simplified record to axiom_scores_daily with a partial_score
    flag in the payload and tier field set correctly.
    """
    try:
        from api.research import build_research_snapshot
        snap = build_research_snapshot(symbol, as_of_date, lookback=60)

        engine_scores: Dict[str, float] = {}

        if "rfs" in engines:
            engine_scores["rfs"] = _compute_rfs_score()

        if "eis" in engines:
            try:
                from api.axiom.engines.fundamental import compute_eis
                fund_domain = snap.get("fundamental_domain") or {}
                eis_inputs = {
                    "cash_flow_durability": (fund_domain.get("positive_fcf_ratio") or 0.5) * 100,
                    "positive_fcf_ratio": fund_domain.get("positive_fcf_ratio"),
                    "gross_margin": (fund_domain.get("latest_quarter") or {}).get("gross_margin"),
                    "revenue_growth_yoy": fund_domain.get("revenue_growth_yoy"),
                }
                engine_scores["eis"] = compute_eis(eis_inputs)
            except Exception:
                engine_scores["eis"] = 50.0

        if "caps" in engines:
            try:
                from api.axiom.engines.fundamental import compute_caps
                fund_domain = snap.get("fundamental_domain") or {}
                lq = (fund_domain.get("latest_quarter") or {})
                caps_inputs = {
                    "return_on_equity": lq.get("op_margin"),
                    "gross_margin": lq.get("gross_margin"),
                    "operating_margin": lq.get("op_margin"),
                    "revenue_growth_yoy": fund_domain.get("revenue_growth_yoy"),
                }
                engine_scores["caps"] = compute_caps(caps_inputs, {})
            except Exception:
                engine_scores["caps"] = 50.0

        weights = {"eis": 0.25, "caps": 0.20, "rfs": 0.15}
        total_weight = sum(weights.get(e, 0.1) for e in engine_scores)
        if total_weight > 0:
            partial_dau = sum(
                engine_scores[e] * weights.get(e, 0.1) for e in engine_scores
            ) / total_weight
        else:
            partial_dau = 50.0

        if db.db_enabled():
            payload = {
                "engine_scores": engine_scores,
                "partial_score": True,
                "engines_run": engines,
                "tier": tier,
                "deployable_alpha_utility": partial_dau,
                "regime_label": _get_regime_label(),
            }
            db.safe_execute(
                """
                INSERT INTO axiom_scores_daily
                    (symbol, as_of_date, deployable_alpha_utility,
                     regime_label, payload, ic_state)
                VALUES (%s, %s, %s, %s, %s::jsonb, 'PARTIAL')
                ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                    deployable_alpha_utility = EXCLUDED.deployable_alpha_utility,
                    regime_label             = EXCLUDED.regime_label,
                    payload                  = EXCLUDED.payload,
                    updated_at               = now()
                """,
                (
                    symbol,
                    as_of_date,
                    partial_dau,
                    payload.get("regime_label", "unknown"),
                    json.dumps(payload),
                ),
            )
        return True

    except Exception as exc:
        logger.debug(
            "_score_partial failed sym=%s engines=%s err=%s", symbol, engines, exc
        )
        return False


def _compute_rfs_score() -> float:
    """
    Get the current RFS (Regime Factor Score) — market-level, same for all symbols.
    Reads from the most recent scored Tier 1 symbol in DB; falls back to 50.
    """
    try:
        if db.db_enabled():
            row = db.safe_fetchone(
                """
                SELECT (payload->>'rfs_score')::float
                FROM axiom_scores_daily
                WHERE as_of_date = (SELECT MAX(as_of_date) FROM axiom_scores_daily)
                  AND payload->>'rfs_score' IS NOT NULL
                LIMIT 1
                """
            )
            if row and row[0]:
                return float(row[0])
    except Exception:
        pass
    return 50.0


def _get_regime_label() -> str:
    """Get the current regime label from the most recent DB score."""
    try:
        if db.db_enabled():
            row = db.safe_fetchone(
                """
                SELECT regime_label FROM axiom_scores_daily
                WHERE as_of_date = (SELECT MAX(as_of_date) FROM axiom_scores_daily)
                  AND regime_label IS NOT NULL
                LIMIT 1
                """
            )
            if row and row[0]:
                return str(row[0])
    except Exception:
        pass
    return "unknown"
