from __future__ import annotations

import datetime as dt
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from api.alpha import (
    CANONICAL_FEATURE_VERSION,
    CANONICAL_SIGNAL_VERSION,
    build_canonical_features,
    build_canonical_signal,
)
from api.research import CANONICAL_SNAPSHOT_VERSION, build_research_snapshot

from .common import (
    BACKTEST_VALIDATION_ARTIFACT_KIND,
    CANONICAL_BACKTEST_VERSION,
    RESEARCH_TRUTH_VERSION,
    WALKFORWARD_VERSION,
    as_date,
    clamp,
    confidence_bucket,
    conviction_tier_from_confidence,
    fragility_tier_from_score,
    horizon_days,
    iso_date,
    mean,
    safe_float,
)
from .cohorts import build_cohort_breakdown
from .outcomes import evaluate_prediction_outcome
from .scorecards import build_failure_modes, build_validation_scorecards
from .walkforward import build_walkforward_validation


def _resolve_horizon_days(label: str) -> int:
    return horizon_days(label)


def _feature_value(record: Dict[str, Any], key: str) -> Optional[float]:
    return safe_float((record.get("feature_vector") or {}).get(key))


def _derive_actionability(score: Optional[float], confidence_score: Optional[float], suppression_count: int) -> float:
    raw_score = abs(safe_float(score) or 0.0)
    confidence = safe_float(confidence_score) or 0.0
    result = raw_score * 55.0 + confidence * 0.55 - suppression_count * 12.0
    return round(clamp(result, 0.0, 100.0), 2)


def _derive_strategy_posture(action: str, actionability: float, suppression_count: int) -> str:
    if action == "BUY":
        if suppression_count >= 2:
            return "watchlist_positive"
        if actionability >= 62:
            return "actionable_long"
        return "trend_continuation_candidate"
    if action == "SELL":
        if suppression_count >= 2:
            return "watchlist_negative"
        if actionability >= 62:
            return "actionable_short"
        return "opportunistic_reversal"
    if suppression_count:
        return "fragile_hold"
    return "wait"


def _derive_deployment_permission(actionability: float, confidence_score: float, suppression_count: int) -> str:
    if suppression_count >= 3 or confidence_score < 28:
        return "analysis_only"
    if suppression_count >= 2 or actionability < 38:
        return "paper_shadow_only"
    if actionability >= 72 and confidence_score >= 70 and suppression_count == 0:
        return "limited_live_eligible"
    return "paper_shadow_only"


def _derive_trust_tier(permission: str) -> str:
    return {
        "limited_live_eligible": "elevated",
        "paper_shadow_only": "developing",
        "analysis_only": "blocked",
    }.get(permission, "blocked")


def _derive_candidate_classification(action: str, actionability: float, suppression_count: int) -> str:
    if action == "HOLD":
        return "watchlist_candidate"
    if suppression_count >= 3:
        return "blocked_candidate"
    if suppression_count >= 2:
        return "too_fragile_candidate"
    if actionability >= 72:
        return "top_priority_candidate"
    if actionability >= 52:
        return "secondary_candidate"
    return "watchlist_candidate"


def _derive_slices(feature_vector: Dict[str, Any], signal_payload: Dict[str, Any]) -> Dict[str, Any]:
    event_state = str(feature_vector.get("event_risk_classification") or "unknown")
    liquidity_state = str(feature_vector.get("tradability_state") or "unknown")
    breadth_state = str(feature_vector.get("breadth_state") or "unknown")
    conflict = safe_float(feature_vector.get("cross_asset_conflict_score")) or 0.0
    stress = safe_float(feature_vector.get("market_stress_score")) or 0.0
    fragility = safe_float(feature_vector.get("implementation_fragility_score")) or 0.0
    return {
        "regime_label": feature_vector.get("signal_regime_label") or signal_payload.get("regime"),
        "event_risk_state": event_state,
        "liquidity_state": liquidity_state,
        "breadth_state": breadth_state,
        "cross_asset_state": "conflicted" if conflict >= 60 else "supportive" if conflict <= 35 else "mixed",
        "stress_state": "unstable" if stress >= 60 else "stable",
        "fragility_tier": fragility_tier_from_score(fragility),
    }


def build_validation_record(
    *,
    symbol: str,
    as_of_date: dt.date,
    horizon: str,
    risk_mode: str,
    signal_result: Dict[str, Any],
) -> Dict[str, Any]:
    signal_payload = dict(signal_result.get("payload") or {})
    feature_payload = dict(signal_result.get("feature_payload") or {})
    feature_vector = dict(signal_result.get("feature_vector") or feature_payload.get("features") or {})
    confidence = safe_float(signal_payload.get("confidence")) or safe_float(signal_result.get("confidence")) or 0.0
    confidence_score = round(confidence * 100.0, 2)
    action = str(signal_result.get("action") or signal_payload.get("signal") or "HOLD").upper()
    suppression_flags = list(signal_payload.get("suppression_flags") or [])
    actionability = _derive_actionability(signal_result.get("score"), confidence_score, len(suppression_flags))
    strategy_posture = _derive_strategy_posture(action, actionability, len(suppression_flags))
    deployment_permission = _derive_deployment_permission(actionability, confidence_score, len(suppression_flags))
    candidate_classification = _derive_candidate_classification(action, actionability, len(suppression_flags))
    slices = _derive_slices(feature_vector, signal_payload)

    return {
        "symbol": symbol,
        "as_of_date": iso_date(as_of_date),
        "horizon": horizon,
        "horizon_days": _resolve_horizon_days(horizon),
        "risk_mode": risk_mode,
        "signal_action": action,
        "final_signal": action,
        "score": safe_float(signal_result.get("score")),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "conviction_tier": conviction_tier_from_confidence(confidence_score),
        "strategy_posture": strategy_posture,
        "actionability_score": actionability,
        "participant_fit": ["swing trader" if horizon == "swing" else "tactical discretionary"],
        "participant_fit_primary": "swing trader" if horizon == "swing" else "tactical discretionary",
        "fragility_tier": slices.get("fragility_tier"),
        "deployment_permission": deployment_permission,
        "trust_tier": _derive_trust_tier(deployment_permission),
        "candidate_classification": candidate_classification,
        "snapshot_id": signal_result.get("snapshot_id"),
        "snapshot_version": signal_result.get("snapshot_version") or CANONICAL_SNAPSHOT_VERSION,
        "feature_version": signal_result.get("feature_version") or CANONICAL_FEATURE_VERSION,
        "signal_version": signal_result.get("signal_version") or CANONICAL_SIGNAL_VERSION,
        "feature_vector": feature_vector,
        "signal_payload": signal_payload,
        "suppression_flags": suppression_flags,
        "slices": slices,
        "regime_label": slices.get("regime_label"),
        "proprietary_scores": signal_result.get("proprietary_scores") or {},
    }


def build_validation_artifact(
    *,
    records: Sequence[Dict[str, Any]],
    cohort_symbol: Optional[str] = None,
    cohort_horizon: Optional[str] = None,
    cohort_risk_mode: Optional[str] = None,
    bar_fetcher: Optional[Callable[[str, dt.date, int], List[Dict[str, Any]]]] = None,
    cost_model: Optional[Dict[str, Any]] = None,
    min_sample_size: int = 6,
    engine_lineage: Optional[Dict[str, Any]] = None,
    walkforward_mode: str = "anchored",
    train_window: int = 252,
    validation_window: int = 63,
    step_window: int = 63,
) -> Dict[str, Any]:
    filtered = [
        dict(record)
        for record in records
        if (cohort_horizon is None or str(record.get("horizon")) == str(cohort_horizon))
        and (cohort_risk_mode is None or str(record.get("risk_mode")) == str(cohort_risk_mode))
        and (cohort_symbol is None or str(record.get("symbol")) == str(cohort_symbol) or cohort_symbol == "*")
    ]
    enriched: List[Dict[str, Any]] = []
    for record in filtered:
        outcome = record.get("outcome")
        if not outcome or (outcome.get("matured") is not True and outcome.get("outcome_status") != "pending"):
            outcome = evaluate_prediction_outcome(
                record,
                bar_fetcher=bar_fetcher,
                cost_model=cost_model,
            )
        enriched.append({**record, "outcome": outcome})

    scorecards = build_validation_scorecards(enriched)
    cohorts = build_cohort_breakdown(enriched)
    failure_modes = build_failure_modes(
        weakest_conditions=cohorts.get("weakest_conditions") or [],
        suppression_effect_summary=scorecards.get("suppression_effect_summary") or {},
    )
    walkforward = build_walkforward_validation(
        enriched,
        mode=walkforward_mode,
        train_window=train_window,
        validation_window=validation_window,
        step_window=step_window,
    )
    matured_count = (scorecards.get("signal_scorecard") or {}).get("final_signal_overall", {}).get("matured_count") or 0
    gross_summary = scorecards.get("gross_return_summary") or {}
    net_summary = scorecards.get("net_return_summary") or {}
    readiness_scorecard = scorecards.get("readiness_scorecard") or {}
    suppression_effect = scorecards.get("suppression_effect_summary") or {}
    mae_mfe = scorecards.get("mae_mfe_summary") or {}

    validation_summary = (
        f"Canonical research truth currently tracks {matured_count} matured point-in-time decisions"
        f" for the {cohort_horizon or 'all-horizon'} / {cohort_risk_mode or 'all-risk'} cohort."
    )
    walkforward_summary = str(
        walkforward.get("calibration_drift_summary")
        or "Walk-forward validation is not yet populated."
    )
    net_friction_summary = (
        f"Gross edge averages {gross_summary.get('average_edge_return')} while net edge averages "
        f"{net_summary.get('average_edge_return')}, with average cost drag "
        f"{(scorecards.get('friction_cost_summary') or {}).get('average_cost_drag')}."
        if gross_summary and net_summary
        else "Net-of-friction validation is not yet populated."
    )
    suppression_readiness_summary = (
        f"Suppression spread is {suppression_effect.get('suppression_effect_edge_spread')}, and paper-vs-live quality spread is "
        f"{readiness_scorecard.get('paper_vs_live_candidate_quality_summary')}."
    )
    drawdown_summary = (
        f"Average MAE is {(mae_mfe.get('mae_distribution') or {}).get('average')}, average MFE is "
        f"{(mae_mfe.get('mfe_distribution') or {}).get('average')}, and invalidation frequency is "
        f"{mae_mfe.get('invalidation_frequency')}."
    )
    status = "available" if matured_count >= min_sample_size else "limited"

    return {
        "artifact_kind": BACKTEST_VALIDATION_ARTIFACT_KIND,
        "validation_version": RESEARCH_TRUTH_VERSION,
        "canonical_backtest_version": CANONICAL_BACKTEST_VERSION,
        "walkforward_version": WALKFORWARD_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "cohort_definition": {
            "symbol": cohort_symbol,
            "horizon": cohort_horizon,
            "risk_mode": cohort_risk_mode,
            "minimum_matured_sample": min_sample_size,
        },
        "engine_lineage": engine_lineage or {},
        "prediction_linkage_summary": {
            "total_predictions": len(enriched),
            "unique_symbols": len({record.get("symbol") for record in enriched if record.get("symbol")}),
            "matured_count": matured_count,
            "earliest_as_of_date": min((record.get("as_of_date") for record in enriched if record.get("as_of_date")), default=None),
            "latest_as_of_date": max((record.get("as_of_date") for record in enriched if record.get("as_of_date")), default=None),
        },
        **scorecards,
        "cohort_breakdown": cohorts.get("cohort_breakdown"),
        "regime_breakdown": cohorts.get("cohort_breakdown"),
        "strongest_conditions": cohorts.get("strongest_conditions"),
        "weakest_conditions": cohorts.get("weakest_conditions"),
        "failure_modes": failure_modes,
        "walkforward_summary": walkforward,
        "walkforward_runs": walkforward.get("windows") or [],
        "validation_summary": validation_summary,
        "walkforward_validation_summary": walkforward_summary,
        "net_of_friction_summary": net_friction_summary,
        "suppression_readiness_validation_summary": suppression_readiness_summary,
        "drawdown_invalidation_summary": drawdown_summary,
    }


def run_canonical_backtest(
    *,
    symbols: Sequence[str],
    bars: Dict[str, Dict[dt.date, float]],
    market_states: Dict[str, Dict[dt.date, Dict[str, float]]],
    start: dt.date,
    end: dt.date,
    horizon: str,
    risk_mode: str,
    cost_model: Dict[str, Any],
    signal_version_hash: str,
    quality_score_fetcher: Callable[[str, dt.date], int],
    signal_resolver: Callable[[str, dt.date], Optional[Dict[str, Any]]],
    friction_engine: Any,
) -> Dict[str, Any]:
    spy_closes = bars.get("SPY", {})
    all_dates = sorted({date for series in bars.values() for date in series.keys() if start <= date <= end})
    positions: Dict[str, Dict[str, Any]] = {}
    trades: List[Dict[str, Any]] = []
    prediction_records: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    daily_returns: List[float] = []
    daily_equity: List[float] = [1.0]
    daily_dates: List[dt.date] = [all_dates[0]]
    trades_notional = 0.0
    signal_cache: Dict[Tuple[str, dt.date], Optional[Dict[str, Any]]] = {}
    lineage: Dict[str, Any] = {
        "snapshot_version": CANONICAL_SNAPSHOT_VERSION,
        "feature_version": CANONICAL_FEATURE_VERSION,
        "signal_version": signal_version_hash,
    }

    for idx, date in enumerate(all_dates[1:], start=1):
        active_returns: List[float] = []
        trade_cost = 0.0
        for sym in symbols:
            closes = bars.get(sym, {})
            if date not in closes:
                continue
            decision = signal_cache.get((sym, date))
            if (sym, date) not in signal_cache:
                decision = signal_resolver(sym, date)
                signal_cache[(sym, date)] = decision
            if decision:
                lineage["snapshot_version"] = decision.get("snapshot_version") or lineage["snapshot_version"]
                lineage["feature_version"] = decision.get("feature_version") or lineage["feature_version"]
                lineage["signal_version"] = decision.get("signal_version") or lineage["signal_version"]
                record = build_validation_record(
                    symbol=sym,
                    as_of_date=date,
                    horizon=horizon,
                    risk_mode=risk_mode,
                    signal_result=decision,
                )
                future_rows = [
                    {
                        "as_of_date": dt_key.isoformat(),
                        **market_states.get(sym, {}).get(dt_key, {"close": closes.get(dt_key)}),
                    }
                    for dt_key in sorted(dt_key for dt_key in closes.keys() if dt_key >= date)[: record["horizon_days"] + 1]
                ]
                record["outcome"] = evaluate_prediction_outcome(
                    record,
                    bar_fetcher=lambda _symbol, _date, _limit, rows=future_rows: rows,
                    cost_model=cost_model,
                    evaluation_as_of_date=end,
                )
                prediction_records.append(record)

            action = str((decision or {}).get("action") or "HOLD").upper() if decision else "HOLD"
            desired = "LONG" if action == "BUY" else "SHORT" if action == "SELL" else None
            current = positions.get(sym)

            if current and desired != current["side"]:
                exit_px = closes[date]
                pnl_pct = (exit_px / current["entry_px"] - 1.0) * (1.0 if current["side"] == "LONG" else -1.0)
                trades.append(
                    {
                        "symbol": sym,
                        "entry_dt": current["entry_dt"],
                        "exit_dt": date,
                        "side": current["side"].lower(),
                        "entry_px": current["entry_px"],
                        "exit_px": exit_px,
                        "qty": current["qty"],
                        "pnl": pnl_pct * current["qty"],
                        "pnl_pct": pnl_pct,
                        "holding_days": (date - current["entry_dt"]).days,
                    }
                )
                positions.pop(sym, None)

            if desired and sym not in positions:
                positions[sym] = {
                    "side": desired,
                    "entry_dt": date,
                    "entry_px": closes[date],
                    "qty": 1.0,
                }
                trades_notional += 1.0

            current = positions.get(sym)
            if current:
                prev_px = closes.get(all_dates[idx - 1])
                if prev_px not in (None, 0):
                    ret = (closes[date] / prev_px - 1.0) * (1.0 if current["side"] == "LONG" else -1.0)
                    active_returns.append(ret)

        portfolio_return = sum(active_returns) / len(active_returns) if active_returns else 0.0
        portfolio_return -= trade_cost / max(daily_equity[-1], 1e-12)
        equity = daily_equity[-1] * (1.0 + portfolio_return)
        daily_returns.append(portfolio_return)
        daily_equity.append(equity)
        daily_dates.append(date)

    benchmark_equity = [1.0]
    for idx in range(1, len(daily_dates)):
        prev = spy_closes.get(daily_dates[idx - 1])
        cur = spy_closes.get(daily_dates[idx])
        if prev in (None, 0) or cur is None:
            benchmark_equity.append(benchmark_equity[-1])
            continue
        benchmark_equity.append(benchmark_equity[-1] * (cur / prev))

    drawdowns: List[float] = []
    peak = daily_equity[0]
    for value in daily_equity:
        peak = max(peak, value)
        drawdowns.append(value / peak - 1.0)
    equity_curve = [
        {
            "dt": daily_dates[index].isoformat(),
            "equity": daily_equity[index],
            "drawdown": drawdowns[index],
            "benchmark_equity": benchmark_equity[index],
        }
        for index in range(len(daily_dates))
    ]

    total_return = daily_equity[-1] - 1.0
    n_days = max(1, len(daily_returns))
    cagr = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    mean_daily = mean(daily_returns) or 0.0
    volatility = (
        math.sqrt(sum((value - mean_daily) ** 2 for value in daily_returns) / max(1, len(daily_returns) - 1)) * math.sqrt(252.0)
        if len(daily_returns) > 1
        else 0.0
    )
    sharpe = (mean_daily / (volatility / math.sqrt(252.0))) if volatility not in (None, 0) else 0.0
    negatives = [value for value in daily_returns if value < 0]
    downside = (
        math.sqrt(sum(value * value for value in negatives) / max(1, len(negatives) - 1)) * math.sqrt(252.0)
        if len(negatives) > 1
        else 0.0
    )
    sortino = (mean_daily / (downside / math.sqrt(252.0))) if downside not in (None, 0) else 0.0
    max_dd = min(drawdowns) if drawdowns else 0.0
    wins = [trade for trade in trades if trade["pnl"] > 0]
    winrate = float(len(wins) / len(trades)) if trades else 0.0
    avg_trade = float(sum(trade["pnl_pct"] for trade in trades) / len(trades)) if trades else 0.0
    years = max(1e-9, n_days / 252.0)
    trades_per_year = float(len(trades) / years) if years > 0 else 0.0
    avg_equity = sum(daily_equity) / len(daily_equity) if daily_equity else 1.0
    turnover = float(trades_notional / avg_equity) if avg_equity else 0.0
    spy_return = benchmark_equity[-1] - 1.0 if benchmark_equity else 0.0

    validation_artifact = build_validation_artifact(
        records=prediction_records,
        cohort_symbol="*" if len(symbols) > 1 else (symbols[0] if symbols else None),
        cohort_horizon=horizon,
        cohort_risk_mode=risk_mode,
        cost_model=cost_model,
        engine_lineage=lineage,
    )
    regime_rows = []
    for item in ((validation_artifact.get("cohort_breakdown") or {}).get("regime_label") or []):
        regime_rows.append(
            {
                "regime_name": item.get("label"),
                "cagr": item.get("average_edge_return") or 0.0,
                "sharpe": item.get("hit_rate") or 0.0,
                "maxdd": -abs(item.get("average_mae") or 0.0),
                "winrate": item.get("hit_rate") or 0.0,
                "trades": item.get("matured_count") or 0,
            }
        )
    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "metrics": {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "maxdd": max_dd,
            "volatility": volatility,
            "winrate": winrate,
            "avgtrade": avg_trade,
            "tradesperyear": trades_per_year,
            "turnover": turnover,
            "alpha_vs_spy": total_return - spy_return,
            "beta": None,
        },
        "regime_metrics": regime_rows,
        "validation_artifact": validation_artifact,
    }


def compute_canonical_signal_for_date(
    symbol: str,
    as_of_date: dt.date,
    *,
    lookback: int = 252,
    quality_score: int = 0,
) -> Optional[Dict[str, Any]]:
    try:
        snapshot = build_research_snapshot(
            symbol,
            as_of_date,
            lookback,
            lookback_days=max(420, lookback * 2),
            include_reference_context=True,
        )
    except Exception:
        return None
    if len(snapshot.get("price_bars") or []) < 30:
        return None
    feature_payload = build_canonical_features(snapshot)
    signal_payload = build_canonical_signal(snapshot, feature_payload, quality_score=quality_score)
    return {
        "action": signal_payload.get("signal"),
        "score": signal_payload.get("score"),
        "confidence": signal_payload.get("confidence"),
        "payload": signal_payload,
        "feature_payload": feature_payload,
        "feature_vector": feature_payload.get("features") or {},
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_version": snapshot.get("snapshot_version"),
        "feature_version": feature_payload.get("feature_version"),
        "signal_version": signal_payload.get("signal_version"),
    }
