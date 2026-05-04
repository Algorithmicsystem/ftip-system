from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from api import db

from .common import as_date, clamp, correlation, covariance, safe_float, stdev


PriceHistoryLoader = Callable[[str, Any, int], List[Dict[str, Any]]]


def _db_ready() -> bool:
    return db.db_enabled() and db.db_read_enabled()


def default_price_history_loader(
    symbol: str,
    as_of_date: Any,
    lookback_days: int = 126,
) -> List[Dict[str, Any]]:
    as_of = as_date(as_of_date)
    if as_of is None or not _db_ready():
        return []
    start = as_of.toordinal() - lookback_days * 2
    start_date = as_of.fromordinal(start)
    try:
        rows = db.safe_fetchall(
            """
            SELECT as_of_date, close
            FROM market_bars_daily
            WHERE symbol = %s
              AND as_of_date >= %s
              AND as_of_date <= %s
            ORDER BY as_of_date ASC
            """,
            (symbol, start_date, as_of),
        )
    except Exception:
        return []
    output: List[Dict[str, Any]] = []
    for row_date, close in rows:
        close_value = safe_float(close)
        row_as_of = as_date(row_date)
        if close_value is None or row_as_of is None:
            continue
        output.append(
            {
                "as_of_date": row_as_of.isoformat(),
                "close": close_value,
            }
        )
    return output


def _returns_from_prices(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[float]]:
    dates: List[str] = []
    returns: List[float] = []
    previous: Optional[float] = None
    for row in rows:
        close = safe_float(row.get("close"))
        date = as_date(row.get("as_of_date"))
        if close is None or date is None:
            previous = close
            continue
        if previous not in (None, 0.0):
            returns.append(float(close / previous - 1.0))
            dates.append(date.isoformat())
        previous = close
    return dates, returns


def _synthetic_history_from_report(report: Dict[str, Any], *, length: int = 84) -> Dict[str, Any]:
    symbol = str(report.get("symbol") or "UNKNOWN")
    bundle = report.get("data_bundle") or {}
    market = bundle.get("market_price_volume") or {}
    canonical = (bundle.get("canonical_alpha_core") or {}).get("feature_vector") or {}
    sector = str((bundle.get("symbol_meta") or {}).get("sector") or (bundle.get("relative_context") or {}).get("sector") or "unknown").lower()
    benchmark = str((bundle.get("relative_context") or {}).get("benchmark_proxy") or (bundle.get("macro_cross_asset") or {}).get("benchmark_proxy") or "SPY").lower()
    theme_items = (bundle.get("sentiment_narrative_flow") or {}).get("top_narratives") or []
    theme = str((theme_items[0] or {}).get("topic") if theme_items and isinstance(theme_items[0], dict) else theme_items[0] if theme_items else "broad_theme").lower()

    seed = sum(ord(char) for char in symbol)
    sector_seed = sum(ord(char) for char in sector)
    benchmark_seed = sum(ord(char) for char in benchmark)
    theme_seed = sum(ord(char) for char in theme)

    ret_21d = safe_float(market.get("ret_21d") or canonical.get("ret_21d")) or 0.0
    realized_vol = safe_float(market.get("realized_vol_21d") or canonical.get("vol_21d")) or 0.22
    fragility = safe_float(
        (bundle.get("liquidity_execution_fragility") or {}).get("implementation_fragility_score")
        or canonical.get("implementation_fragility_score")
        or report.get("signal_fragility_index")
    ) or 45.0
    crowding = safe_float(report.get("narrative_crowding_index")) or 40.0
    macro_alignment = safe_float(report.get("macro_alignment_score")) or 50.0
    event_overhang = safe_float(
        (bundle.get("event_catalyst_risk") or {}).get("event_overhang_score")
        or canonical.get("event_overhang_score")
    ) or 32.0

    drift = clamp(ret_21d / 21.0, -0.012, 0.012)
    daily_vol = clamp(realized_vol / math.sqrt(252.0), 0.004, 0.04)
    macro_bias = (macro_alignment - 50.0) / 9000.0
    fragility_drag = max(0.0, fragility - 50.0) / 18000.0
    crowding_drag = max(0.0, crowding - 55.0) / 22000.0
    event_drag = max(0.0, event_overhang - 50.0) / 18000.0

    dates: List[str] = []
    returns: List[float] = []
    base_date = as_date(report.get("as_of_date"))
    for offset in range(length, 0, -1):
        day_index = length - offset
        anchor = day_index + 1
        sector_wave = math.sin((anchor + sector_seed % 9) / 5.0) * daily_vol * 0.34
        benchmark_wave = math.cos((anchor + benchmark_seed % 11) / 7.0) * daily_vol * 0.28
        theme_wave = math.sin((anchor + theme_seed % 13) / 9.0) * daily_vol * 0.16
        idio_wave = math.sin((anchor + seed % 17) / 4.0) * daily_vol * 0.22
        shock = 0.0
        if fragility >= 55.0 and anchor % 17 == seed % 17:
            shock -= daily_vol * (0.45 + fragility / 250.0)
        if event_overhang >= 65.0 and anchor % 21 == theme_seed % 21:
            shock -= daily_vol * 0.35
        returns.append(
            drift
            + macro_bias
            + sector_wave
            + benchmark_wave
            + theme_wave
            + idio_wave
            - fragility_drag
            - crowding_drag
            - event_drag
            + shock
        )
        if base_date is not None:
            dates.append((base_date.fromordinal(base_date.toordinal() - offset)).isoformat())
        else:
            dates.append(f"t-{offset}")
    return {
        "dates": dates,
        "returns": returns,
        "history_source": "synthetic_proxy",
        "available_days": len(returns),
        "relationship_confidence": 34.0,
        "insufficient_history": True,
    }


def build_return_profile(
    report: Dict[str, Any],
    *,
    history_loader: Optional[PriceHistoryLoader] = None,
    lookback_days: int = 126,
) -> Dict[str, Any]:
    bundle = report.get("data_bundle") or {}
    for key in (
        "return_history",
        "trailing_return_history",
        "price_history",
    ):
        embedded = (bundle.get("portfolio_risk_inputs") or {}).get(key)
        if embedded:
            dates, returns = _returns_from_prices(embedded)
            if len(returns) >= 20:
                return {
                    "dates": dates[-lookback_days:],
                    "returns": returns[-lookback_days:],
                    "history_source": "embedded_history",
                    "available_days": min(len(returns), lookback_days),
                    "relationship_confidence": 76.0,
                    "insufficient_history": len(returns) < 63,
                }

    loader = history_loader or default_price_history_loader
    rows = loader(str(report.get("symbol") or ""), report.get("as_of_date"), lookback_days)
    dates, returns = _returns_from_prices(rows)
    if len(returns) >= 20:
        confidence = clamp(45.0 + min(len(returns), lookback_days) / lookback_days * 45.0, 45.0, 90.0)
        return {
            "dates": dates[-lookback_days:],
            "returns": returns[-lookback_days:],
            "history_source": "realized_history",
            "available_days": min(len(returns), lookback_days),
            "relationship_confidence": round(confidence, 2),
            "insufficient_history": len(returns) < 63,
        }
    return _synthetic_history_from_report(report)


def pairwise_relationship(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Any]:
    left_map = {
        date: float(value)
        for date, value in zip(left.get("dates") or [], left.get("returns") or [])
    }
    right_map = {
        date: float(value)
        for date, value in zip(right.get("dates") or [], right.get("returns") or [])
    }
    overlap_dates = sorted(set(left_map.keys()) & set(right_map.keys()))
    if len(overlap_dates) < 8:
        return {
            "pairwise_correlation": None,
            "pairwise_covariance": None,
            "rolling_correlation_profile": {},
            "correlation_stability_score": 28.0,
            "covariance_cluster_score": 26.0,
            "relationship_confidence": min(
                float(left.get("relationship_confidence") or 30.0),
                float(right.get("relationship_confidence") or 30.0),
            ),
            "insufficient_history": True,
            "stress_correlation_adjustment": 0.0,
            "correlation_breakdown_risk": 62.0,
        }

    left_returns = [left_map[date] for date in overlap_dates]
    right_returns = [right_map[date] for date in overlap_dates]
    full_corr = correlation(left_returns, right_returns)
    full_cov = covariance(left_returns, right_returns)
    short_corr = correlation(left_returns[-21:], right_returns[-21:]) if len(left_returns) >= 21 else full_corr
    medium_corr = correlation(left_returns[-63:], right_returns[-63:]) if len(left_returns) >= 63 else full_corr
    diff = abs((short_corr or 0.0) - (medium_corr or 0.0))
    stability = round(clamp(100.0 - diff * 140.0, 0.0, 100.0), 2)

    stress_pairs = [
        (lret, rret)
        for lret, rret in zip(left_returns, right_returns)
        if abs(lret) >= max(stdev(left_returns[-63:] or left_returns) or 0.0, 0.012)
        or abs(rret) >= max(stdev(right_returns[-63:] or right_returns) or 0.0, 0.012)
    ]
    stress_corr = (
        correlation([item[0] for item in stress_pairs], [item[1] for item in stress_pairs])
        if len(stress_pairs) >= 5
        else full_corr
    )
    stress_adjustment = round(
        clamp(abs(stress_corr or 0.0) * 100.0 - abs(full_corr or 0.0) * 100.0, -50.0, 50.0),
        2,
    )
    cov_cluster = round(
        clamp(
            abs(full_corr or 0.0) * 58.0
            + max(0.0, abs(stress_corr or 0.0) - 0.35) * 42.0
            + (100.0 - stability) * 0.18,
            0.0,
            100.0,
        ),
        2,
    )
    confidence = round(
        clamp(
            min(
                float(left.get("relationship_confidence") or 30.0),
                float(right.get("relationship_confidence") or 30.0),
            )
            * min(len(overlap_dates) / 63.0, 1.0),
            18.0,
            92.0,
        ),
        2,
    )
    return {
        "pairwise_correlation": round(full_corr or 0.0, 6) if full_corr is not None else None,
        "pairwise_covariance": round(full_cov or 0.0, 8) if full_cov is not None else None,
        "rolling_correlation_profile": {
            "short_window": round(short_corr or 0.0, 6) if short_corr is not None else None,
            "medium_window": round(medium_corr or 0.0, 6) if medium_corr is not None else None,
            "stress_window": round(stress_corr or 0.0, 6) if stress_corr is not None else None,
        },
        "correlation_stability_score": stability,
        "covariance_cluster_score": cov_cluster,
        "relationship_confidence": confidence,
        "insufficient_history": len(overlap_dates) < 21
        or bool(left.get("insufficient_history"))
        or bool(right.get("insufficient_history")),
        "stress_correlation_adjustment": stress_adjustment,
        "correlation_breakdown_risk": round(clamp((100.0 - stability) * 0.74 + max(0.0, stress_adjustment), 0.0, 100.0), 2),
        "overlap_days": len(overlap_dates),
    }
