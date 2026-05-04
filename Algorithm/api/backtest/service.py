from __future__ import annotations

import datetime as dt
import math
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from api.alpha import CANONICAL_SIGNAL_VERSION, build_canonical_features, build_canonical_signal
from api import config, db
from api.data_providers import canonical_symbol
from api.research import (
    BACKTEST_VALIDATION_ARTIFACT_KIND,
    build_research_snapshot,
    run_canonical_backtest as run_canonical_backtest_engine,
)
from ftip.friction import CostModel as FrictionCostModel
from ftip.friction import ExecutionPlan, FrictionEngine, MarketStateInputs


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _resolve_symbols(symbol: Optional[str], universe: str) -> List[str]:
    if symbol:
        return [canonical_symbol(symbol)]
    rows = db.safe_fetchall("SELECT symbol FROM market_symbols WHERE is_active = TRUE")
    return [row[0] for row in rows]


def _fetch_bars(
    symbols: List[str], start: dt.date, end: dt.date
) -> Dict[str, Dict[dt.date, float]]:
    if not symbols:
        return {}
    rows = db.safe_fetchall(
        """
        SELECT symbol, as_of_date, close
        FROM market_bars_daily
        WHERE symbol = ANY(%s)
          AND as_of_date BETWEEN %s AND %s
        ORDER BY symbol, as_of_date
        """,
        (symbols, start, end),
    )
    out: Dict[str, Dict[dt.date, float]] = {}
    for sym, as_of_date, close in rows:
        out.setdefault(sym, {})[as_of_date] = float(close)
    return out


def _fetch_market_states(
    symbols: List[str], start: dt.date, end: dt.date
) -> Dict[str, Dict[dt.date, Dict[str, float]]]:
    if not symbols:
        return {}
    rows = db.safe_fetchall(
        """
        SELECT symbol, as_of_date, open, high, low, close, volume
        FROM market_bars_daily
        WHERE symbol = ANY(%s)
          AND as_of_date BETWEEN %s AND %s
        ORDER BY symbol, as_of_date
        """,
        (symbols, start, end),
    )
    out: Dict[str, Dict[dt.date, Dict[str, float]]] = {}
    for symbol, as_of_date, open_px, high_px, low_px, close_px, volume in rows:
        out.setdefault(symbol, {})[as_of_date] = {
            "open": float(open_px or close_px),
            "high": float(high_px or close_px),
            "low": float(low_px or close_px),
            "close": float(close_px),
            "volume": float(volume or 0.0),
        }
    return out


def _fetch_signal(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT action, score, confidence
        FROM signals_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        return None
    return {"action": row[0], "score": row[1], "confidence": row[2]}


def _fetch_features(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT ret_21d, vol_63d, atr_pct, mom_vol_adj_21d, maxdd_63d
        FROM features_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not row:
        return None
    return {
        "ret_21d": row[0],
        "vol_63d": row[1],
        "atr_pct": row[2],
        "mom_vol_adj_21d": row[3],
        "maxdd_63d": row[4],
    }


def _fetch_quality_score(symbol: str, as_of_date: dt.date) -> int:
    row = db.safe_fetchone(
        """
        SELECT quality_score
        FROM quality_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    return int(row[0]) if row and row[0] is not None else 0


def _compute_signal_if_missing(
    symbol: str, as_of_date: dt.date
) -> Optional[Dict[str, Any]]:
    signal = _compute_canonical_signal_for_date(symbol, as_of_date)
    if not signal:
        return None
    return {
        "action": signal["action"],
        "score": signal["score"],
        "confidence": signal["confidence"],
    }


def _compute_canonical_signal_for_date(
    symbol: str, as_of_date: dt.date, lookback: int = 252
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
    signal_payload = build_canonical_signal(
        snapshot,
        feature_payload,
        quality_score=_fetch_quality_score(symbol, as_of_date),
    )
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


def _signal_for_date(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    return _compute_signal_if_missing(symbol, as_of_date)


def _daily_returns(
    closes: Dict[dt.date, float], dates: List[dt.date]
) -> Dict[dt.date, float]:
    out: Dict[dt.date, float] = {}
    for i in range(1, len(dates)):
        prev = closes.get(dates[i - 1])
        cur = closes.get(dates[i])
        if prev is None or cur is None or prev == 0:
            continue
        out[dates[i]] = float(cur / prev - 1.0)
    return out


def _max_drawdown(equity: List[float]) -> float:
    peak = equity[0] if equity else 1.0
    max_dd = 0.0
    for value in equity:
        if value > peak:
            peak = value
        dd = (value / peak) - 1.0
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return float(mean / std * math.sqrt(252.0))


def _sortino(returns: List[float]) -> float:
    negatives = [r for r in returns if r < 0]
    if len(negatives) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r) ** 2 for r in negatives) / (len(negatives) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return float(mean / std * math.sqrt(252.0))


def _regime_labels(
    spy_dates: List[dt.date], spy_closes: Dict[dt.date, float]
) -> Dict[dt.date, str]:
    dates_sorted = sorted(spy_dates)
    returns = _daily_returns(spy_closes, dates_sorted)
    vol_values: List[float] = []
    labels: Dict[dt.date, str] = {}
    rolling_returns: List[float] = []
    rolling_closes: List[float] = []

    for idx, date in enumerate(dates_sorted):
        if idx > 0 and date in returns:
            rolling_returns.append(returns[date])
        if date in spy_closes:
            rolling_closes.append(spy_closes[date])

        if len(rolling_returns) > 20:
            rolling_returns.pop(0)
        if len(rolling_closes) > 40:
            rolling_closes.pop(0)

        if len(rolling_returns) < 10 or len(rolling_closes) < 20:
            labels[date] = "UNKNOWN"
            continue

        vol = math.sqrt(sum(r * r for r in rolling_returns) / len(rolling_returns))
        vol_values.append(vol)
        median_vol = sorted(vol_values)[len(vol_values) // 2]

        sma_short = sum(rolling_closes[-10:]) / 10.0
        sma_long = sum(rolling_closes[-20:]) / 20.0
        slope = sma_short - sma_long

        trend_label = "TREND" if slope > 0 else "RANGE"
        vol_label = "HIGH_VOL" if vol >= median_vol else "LOW_VOL"
        labels[date] = f"{trend_label}_{vol_label}"

    return labels


def _bench_equity(
    spy_closes: Dict[dt.date, float], dates: List[dt.date]
) -> List[float]:
    equity = [1.0]
    for i in range(1, len(dates)):
        prev = spy_closes.get(dates[i - 1])
        cur = spy_closes.get(dates[i])
        if prev is None or cur is None or prev == 0:
            equity.append(equity[-1])
            continue
        equity.append(equity[-1] * (cur / prev))
    return equity


def run_backtest(
    *,
    symbol: Optional[str],
    universe: str,
    date_start: str,
    date_end: str,
    horizon: str,
    risk_mode: str,
    signal_version_hash: str,
    cost_model: Dict[str, Any],
) -> Dict[str, Any]:
    _require_db_enabled(write=True, read=True)
    start = _parse_date(date_start)
    end = _parse_date(date_end)
    if start > end:
        raise HTTPException(status_code=400, detail="date_start must be <= date_end")

    signal_version = signal_version_hash
    if signal_version_hash == "auto":
        signal_version = config.env("FTIP_SIGNAL_VERSION_HASH", CANONICAL_SIGNAL_VERSION) or CANONICAL_SIGNAL_VERSION

    symbols = _resolve_symbols(symbol, universe)
    if not symbols:
        raise HTTPException(status_code=400, detail="no symbols available for backtest")

    max_symbols = config.env_int("FTIP_BACKTEST_MAX_SYMBOLS", 25)
    symbols = symbols[:max_symbols]

    bars = _fetch_bars(symbols + ["SPY"], start, end)
    if not bars:
        raise HTTPException(status_code=404, detail="no bar data found")

    spy_closes = bars.get("SPY", {})
    all_dates = sorted(
        {d for series in bars.values() for d in series.keys() if start <= d <= end}
    )
    if len(all_dates) < 2:
        raise HTTPException(
            status_code=400, detail="not enough data points for backtest"
        )

    friction_cost_model = FrictionCostModel.from_legacy(cost_model)
    friction_engine = FrictionEngine(friction_cost_model)
    market_states = _fetch_market_states(symbols, start, end)
    run_payload = run_canonical_backtest_engine(
        symbols=symbols,
        bars=bars,
        market_states=market_states,
        start=start,
        end=end,
        horizon=horizon,
        risk_mode=risk_mode,
        cost_model=cost_model,
        signal_version_hash=signal_version,
        quality_score_fetcher=_fetch_quality_score,
        signal_resolver=_signal_for_date,
        friction_engine=friction_engine,
    )

    run_id = str(uuid.uuid4())
    _persist_backtest(
        run_id=run_id,
        symbol=symbol,
        universe=universe,
        date_start=start,
        date_end=end,
        horizon=horizon,
        risk_mode=risk_mode,
        signal_version_hash=signal_version,
        cost_model=cost_model,
        status="success",
        trades=run_payload["trades"],
        equity_curve=run_payload["equity_curve"],
        metrics=run_payload["metrics"],
        regime_metrics=run_payload["regime_metrics"],
    )
    _persist_backtest_artifact(
        run_id=run_id,
        kind=BACKTEST_VALIDATION_ARTIFACT_KIND,
        payload=run_payload["validation_artifact"],
    )

    return {"run_id": run_id, "status": "success"}


def _persist_backtest(
    *,
    run_id: str,
    symbol: Optional[str],
    universe: str,
    date_start: dt.date,
    date_end: dt.date,
    horizon: str,
    risk_mode: str,
    signal_version_hash: str,
    cost_model: Dict[str, Any],
    status: str,
    trades: List[Dict[str, Any]],
    equity_curve: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    regime_metrics: List[Dict[str, Any]],
    error_message: Optional[str] = None,
) -> None:
    _require_db_enabled(write=True, read=True)
    with db.with_connection() as (conn, cur):
        cur.execute(
            """
            INSERT INTO backtest_runs (
                id, universe, symbol, date_start, date_end, horizon, risk_mode, signal_version_hash,
                cost_model, status, error_message
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
            """,
            (
                run_id,
                universe,
                symbol,
                date_start,
                date_end,
                horizon,
                risk_mode,
                signal_version_hash,
                cost_model,
                status,
                error_message,
            ),
        )

        for trade in trades:
            cur.execute(
                """
                INSERT INTO backtest_trades (
                    run_id, symbol, entry_dt, exit_dt, side, entry_px, exit_px, qty,
                    pnl, pnl_pct, holding_days
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    trade["symbol"],
                    trade["entry_dt"],
                    trade["exit_dt"],
                    trade["side"],
                    trade["entry_px"],
                    trade["exit_px"],
                    trade["qty"],
                    trade["pnl"],
                    trade["pnl_pct"],
                    trade["holding_days"],
                ),
            )

        for point in equity_curve:
            cur.execute(
                """
                INSERT INTO backtest_equity_curve (
                    run_id, dt, equity, drawdown, benchmark_equity
                ) VALUES (%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    dt.date.fromisoformat(point["dt"]),
                    point["equity"],
                    point["drawdown"],
                    point["benchmark_equity"],
                ),
            )

        cur.execute(
            """
            INSERT INTO backtest_metrics (
                run_id, cagr, sharpe, sortino, maxdd, volatility, winrate, avgtrade,
                tradesperyear, turnover, alpha_vs_spy, beta
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id,
                metrics["cagr"],
                metrics["sharpe"],
                metrics["sortino"],
                metrics["maxdd"],
                metrics["volatility"],
                metrics["winrate"],
                metrics["avgtrade"],
                metrics["tradesperyear"],
                metrics["turnover"],
                metrics.get("alpha_vs_spy"),
                metrics.get("beta"),
            ),
        )

        for reg in regime_metrics:
            cur.execute(
                """
                INSERT INTO backtest_regime_metrics (
                    run_id, regime_name, cagr, sharpe, maxdd, winrate, trades
                ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    reg["regime_name"],
                    reg["cagr"],
                    reg["sharpe"],
                    reg["maxdd"],
                    reg["winrate"],
                    reg["trades"],
                ),
            )
        conn.commit()


def _persist_backtest_artifact(*, run_id: str, kind: str, payload: Dict[str, Any]) -> None:
    _require_db_enabled(write=True, read=True)
    with db.with_connection() as (conn, cur):
        cur.execute(
            """
            INSERT INTO backtest_artifacts (run_id, kind, payload)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (run_id, kind)
            DO UPDATE SET payload=EXCLUDED.payload, updated_at=now()
            """,
            (run_id, kind, payload),
        )
        conn.commit()


def fetch_results(run_id: str) -> Dict[str, Any]:
    _require_db_enabled(read=True)
    run_row = db.safe_fetchone(
        """
        SELECT id, created_at, universe, symbol, date_start, date_end, horizon, risk_mode,
               signal_version_hash, cost_model, status, error_message
        FROM backtest_runs
        WHERE id = %s
        """,
        (run_id,),
    )
    if not run_row:
        raise HTTPException(status_code=404, detail="run not found")

    metrics_row = db.safe_fetchone(
        """
        SELECT cagr, sharpe, sortino, maxdd, volatility, winrate, avgtrade, tradesperyear, turnover,
               alpha_vs_spy, beta
        FROM backtest_metrics
        WHERE run_id = %s
        """,
        (run_id,),
    )
    regime_rows = db.safe_fetchall(
        """
        SELECT regime_name, cagr, sharpe, maxdd, winrate, trades
        FROM backtest_regime_metrics
        WHERE run_id = %s
        ORDER BY regime_name
        """,
        (run_id,),
    )
    artifact_rows = db.safe_fetchall(
        """
        SELECT kind, payload
        FROM backtest_artifacts
        WHERE run_id = %s
        ORDER BY kind
        """,
        (run_id,),
    )

    metrics = {
        "cagr": metrics_row[0] if metrics_row else None,
        "sharpe": metrics_row[1] if metrics_row else None,
        "sortino": metrics_row[2] if metrics_row else None,
        "maxdd": metrics_row[3] if metrics_row else None,
        "volatility": metrics_row[4] if metrics_row else None,
        "winrate": metrics_row[5] if metrics_row else None,
        "avgtrade": metrics_row[6] if metrics_row else None,
        "tradesperyear": metrics_row[7] if metrics_row else None,
        "turnover": metrics_row[8] if metrics_row else None,
        "alpha_vs_spy": metrics_row[9] if metrics_row else None,
        "beta": metrics_row[10] if metrics_row else None,
    }
    regime_metrics = [
        {
            "regime_name": row[0],
            "cagr": row[1],
            "sharpe": row[2],
            "maxdd": row[3],
            "winrate": row[4],
            "trades": row[5],
        }
        for row in regime_rows
    ]

    best_regime = None
    worst_regime = None
    if regime_metrics:
        sorted_regimes = sorted(
            regime_metrics, key=lambda item: item.get("cagr") or 0.0, reverse=True
        )
        best_regime = sorted_regimes[0]["regime_name"]
        worst_regime = sorted_regimes[-1]["regime_name"]

    summary = {
        "beats_spy": metrics.get("alpha_vs_spy", 0) is not None
        and metrics.get("alpha_vs_spy", 0) > 0,
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "turnover": metrics.get("turnover"),
    }
    artifacts = {row[0]: row[1] for row in artifact_rows}
    canonical_validation = artifacts.get(BACKTEST_VALIDATION_ARTIFACT_KIND)
    if canonical_validation:
        summary["walkforward_windows"] = (
            canonical_validation.get("walkforward_summary") or {}
        ).get("window_count")
        summary["net_edge_return"] = (
            canonical_validation.get("net_return_summary") or {}
        ).get("average_edge_return")

    return {
        "run": {
            "id": run_row[0],
            "created_at": run_row[1].isoformat() if run_row[1] else None,
            "universe": run_row[2],
            "symbol": run_row[3],
            "date_start": run_row[4].isoformat() if run_row[4] else None,
            "date_end": run_row[5].isoformat() if run_row[5] else None,
            "horizon": run_row[6],
            "risk_mode": run_row[7],
            "signal_version_hash": run_row[8],
            "cost_model": run_row[9],
            "status": run_row[10],
            "error_message": run_row[11],
        },
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "artifacts": artifacts,
        "canonical_validation": canonical_validation,
        "summary": summary,
    }


def fetch_equity_curve(run_id: str) -> List[Dict[str, Any]]:
    _require_db_enabled(read=True)
    rows = db.safe_fetchall(
        """
        SELECT dt, equity, drawdown, benchmark_equity
        FROM backtest_equity_curve
        WHERE run_id = %s
        ORDER BY dt
        """,
        (run_id,),
    )
    return [
        {
            "dt": row[0].isoformat(),
            "equity": row[1],
            "drawdown": row[2],
            "benchmark_equity": row[3],
        }
        for row in rows
    ]
