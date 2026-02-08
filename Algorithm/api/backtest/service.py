from __future__ import annotations

import datetime as dt
import math
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from api import config, db
from api.data_providers import canonical_symbol


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
    from api.signal_engine import compute_daily_signal

    features = _fetch_features(symbol, as_of_date)
    if not features:
        return None
    bars = db.safe_fetchone(
        """
        SELECT close
        FROM market_bars_daily
        WHERE symbol = %s AND as_of_date = %s
        """,
        (symbol, as_of_date),
    )
    if not bars:
        return None
    quality_score = _fetch_quality_score(symbol, as_of_date)
    signal = compute_daily_signal(features, quality_score, float(bars[0]))
    return {
        "action": signal["action"],
        "score": signal["score"],
        "confidence": signal["confidence"],
    }


def _signal_for_date(symbol: str, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    signal = _fetch_signal(symbol, as_of_date)
    if signal:
        return signal
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
        signal_version = config.env("FTIP_SIGNAL_VERSION_HASH", "auto") or "auto"

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

    cost_rate = (
        float(cost_model.get("fee_bps", 0)) + float(cost_model.get("slippage_bps", 0))
    ) / 10000.0
    positions: Dict[str, Dict[str, Any]] = {}
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []

    equity = 1.0
    daily_returns: List[float] = []
    daily_equity: List[float] = [equity]
    daily_dates: List[dt.date] = [all_dates[0]]
    trades_notional = 0.0

    for i, date in enumerate(all_dates[1:], start=1):
        active_returns: List[float] = []
        trade_cost = 0.0

        for sym in symbols:
            closes = bars.get(sym, {})
            if date not in closes:
                continue
            signal = _signal_for_date(sym, date)
            desired = None
            if signal:
                action = (signal.get("action") or "").upper()
                if action == "BUY":
                    desired = "LONG"
                elif action == "SELL":
                    desired = "SHORT"
            current = positions.get(sym)

            if current and (desired != current["side"]):
                exit_px = closes[date]
                pnl_pct = (exit_px / current["entry_px"] - 1.0) * (
                    1.0 if current["side"] == "LONG" else -1.0
                )
                pnl = pnl_pct * current["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry_dt": current["entry_dt"],
                        "exit_dt": date,
                        "side": current["side"].lower(),
                        "entry_px": current["entry_px"],
                        "exit_px": exit_px,
                        "qty": current["qty"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "holding_days": (date - current["entry_dt"]).days,
                    }
                )
                trade_cost += cost_rate
                trades_notional += current["qty"]
                positions.pop(sym, None)

            if desired and sym not in positions:
                positions[sym] = {
                    "side": desired,
                    "entry_dt": date,
                    "entry_px": closes[date],
                    "qty": 1.0,
                }
                trade_cost += cost_rate
                trades_notional += 1.0

            current = positions.get(sym)
            if current:
                prev_px = closes.get(all_dates[i - 1])
                if prev_px:
                    ret = (closes[date] / prev_px - 1.0) * (
                        1.0 if current["side"] == "LONG" else -1.0
                    )
                    active_returns.append(ret)

        portfolio_return = (
            sum(active_returns) / len(active_returns) if active_returns else 0.0
        )
        portfolio_return -= trade_cost
        equity *= 1.0 + portfolio_return
        daily_returns.append(portfolio_return)
        daily_equity.append(equity)
        daily_dates.append(date)

    benchmark_equity = _bench_equity(spy_closes, daily_dates)
    drawdowns = []
    peak = daily_equity[0] if daily_equity else 1.0
    for value in daily_equity:
        if value > peak:
            peak = value
        drawdowns.append(value / peak - 1.0)

    equity_curve = [
        {
            "dt": daily_dates[i].isoformat(),
            "equity": daily_equity[i],
            "drawdown": drawdowns[i],
            "benchmark_equity": (
                benchmark_equity[i]
                if i < len(benchmark_equity)
                else benchmark_equity[-1]
            ),
        }
        for i in range(len(daily_dates))
    ]

    total_return = daily_equity[-1] - 1.0
    n_days = max(1, len(daily_returns))
    cagr = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    volatility = (
        math.sqrt(
            sum((r - (sum(daily_returns) / n_days)) ** 2 for r in daily_returns)
            / max(1, n_days - 1)
        )
        * math.sqrt(252.0)
        if len(daily_returns) > 1
        else 0.0
    )
    sharpe = _sharpe(daily_returns)
    sortino = _sortino(daily_returns)
    max_dd = _max_drawdown(daily_equity)
    wins = [t for t in trades if t["pnl"] > 0]
    winrate = float(len(wins) / len(trades)) if trades else 0.0
    avg_trade = (
        float(sum(t["pnl_pct"] for t in trades) / len(trades)) if trades else 0.0
    )
    years = max(1e-9, n_days / 252.0)
    trades_per_year = float(len(trades) / years) if years > 0 else 0.0
    avg_equity = sum(daily_equity) / len(daily_equity) if daily_equity else 1.0
    turnover = float(trades_notional / avg_equity) if avg_equity else 0.0

    spy_return = benchmark_equity[-1] - 1.0 if benchmark_equity else 0.0
    alpha_vs_spy = total_return - spy_return

    regime_labels = _regime_labels(daily_dates, spy_closes)
    regime_metrics: Dict[str, Dict[str, Any]] = {}
    for idx, date in enumerate(daily_dates[1:], start=1):
        label = regime_labels.get(date, "UNKNOWN")
        bucket = regime_metrics.setdefault(label, {"returns": [], "trades": 0})
        bucket["returns"].append(daily_returns[idx - 1])
    for trade in trades:
        label = regime_labels.get(trade["exit_dt"], "UNKNOWN")
        regime_metrics.setdefault(label, {"returns": [], "trades": 0})
        regime_metrics[label]["trades"] += 1

    regime_rows: List[Dict[str, Any]] = []
    for label, data in regime_metrics.items():
        rets = data["returns"]
        if not rets:
            continue
        r_cagr = (
            (1.0 + sum(rets)) ** (252.0 / len(rets)) - 1.0 if len(rets) > 0 else 0.0
        )
        r_sharpe = _sharpe(rets)
        r_maxdd = min(0.0, min(rets)) if rets else 0.0
        r_winrate = float(len([r for r in rets if r > 0]) / len(rets)) if rets else 0.0
        regime_rows.append(
            {
                "regime_name": label,
                "cagr": r_cagr,
                "sharpe": r_sharpe,
                "maxdd": r_maxdd,
                "winrate": r_winrate,
                "trades": data["trades"],
            }
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
        trades=trades,
        equity_curve=equity_curve,
        metrics={
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "maxdd": max_dd,
            "volatility": volatility,
            "winrate": winrate,
            "avgtrade": avg_trade,
            "tradesperyear": trades_per_year,
            "turnover": turnover,
            "alpha_vs_spy": alpha_vs_spy,
            "beta": None,
        },
        regime_metrics=regime_rows,
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
