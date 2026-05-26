"""Session 20: AXIOM backtest harness.

POST /jobs/axiom-backtest/run   — run backtest over a historical date range
GET  /jobs/axiom-backtest/runs  — list recent stored backtest runs

Pairs stored axiom_scores_daily signals with prosperity_daily_bars prices
to compute hit rate, Sharpe, max drawdown, Spearman IC, and an equity curve.
No AXIOM recomputation — uses existing stored scores only.
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

_SIGNAL_QUERY_LIMIT = 10_000


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def _find_horizon_price(
    date_price_map: Dict[Any, float],
    signal_date: dt.date,
    horizon_days: int,
) -> Optional[float]:
    """Return the closest available close price at or after signal_date + horizon_days."""
    target = signal_date + dt.timedelta(days=horizon_days)
    for delta in range(8):
        candidate = target + dt.timedelta(days=delta)
        if candidate in date_price_map:
            return date_price_map[candidate]
    return None


# ---------------------------------------------------------------------------
# Core stats computation (pure — no DB)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak, mdd = equity[0], 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def compute_backtest_stats(
    raw_signals: List[Dict[str, Any]],
    price_map: Dict[str, Dict[Any, float]],
    horizon_days: int,
    signal_filter: List[str],
) -> Dict[str, Any]:
    """
    Core computation. raw_signals is a list of dicts with keys:
      symbol, as_of_date, regime_label, dau, signal_label.
    price_map is {symbol: {date: close}}.

    Returns full stats dict with equity_curve, by_regime, etc.
    """
    EMPTY = {
        "total_signals": 0, "signals_with_return": 0,
        "hit_rate": None, "avg_return_pct": None,
        "sharpe": None, "max_drawdown": None, "spearman_ic": None,
        "by_regime": {}, "by_signal": {},
        "equity_curve": {}, "portfolio_returns": [],
    }
    if not raw_signals:
        return EMPTY

    results: List[Dict[str, Any]] = []

    for sig in raw_signals:
        signal_label = sig.get("signal_label") or "HOLD"
        if signal_label not in signal_filter:
            continue

        sym = sig["symbol"]
        signal_date = sig["as_of_date"]
        dau = sig.get("dau")

        sym_prices = price_map.get(sym, {})
        p0 = sym_prices.get(signal_date)
        p1 = _find_horizon_price(sym_prices, signal_date, horizon_days)

        if p0 is None or p1 is None or p0 <= 0:
            continue

        raw_ret = (p1 / p0) - 1.0
        # Direction-adjusted: BUY long, SELL short
        adj_ret = raw_ret if signal_label == "BUY" else -raw_ret
        hit = adj_ret > 0

        results.append({
            "symbol": sym,
            "signal_date": signal_date,
            "signal_label": signal_label,
            "dau": dau,
            "regime_label": sig.get("regime_label") or "unknown",
            "raw_return": raw_ret,
            "adj_return": adj_ret,
            "hit": hit,
        })

    if not results:
        return EMPTY

    n = len(results)
    hits = sum(1 for r in results if r["hit"])
    adj_returns = [r["adj_return"] for r in results]

    # Equity curve: group by signal_date, mean adj_return per date
    by_date: Dict[Any, List[float]] = defaultdict(list)
    for r in results:
        by_date[r["signal_date"]].append(r["adj_return"])

    sorted_dates = sorted(by_date.keys())
    port_returns = [_mean(by_date[d]) for d in sorted_dates]

    equity = [1.0]
    for ret in port_returns:
        equity.append(equity[-1] * (1.0 + ret))

    # Sharpe (annualised from H-day periods)
    periods_per_year = 252.0 / max(horizon_days, 1)
    sharpe: Optional[float] = None
    if len(port_returns) >= 2:
        mu = _mean(port_returns)
        sd = _std(port_returns)
        sharpe = round(mu / sd * (periods_per_year ** 0.5), 4) if sd > 0 else 0.0

    mdd = round(_max_drawdown(equity[1:] or equity), 4)

    # Spearman IC: DAU vs direction-adjusted return
    ic: Optional[float] = None
    ic_pairs = [(r["dau"], r["adj_return"]) for r in results if r["dau"] is not None]
    if len(ic_pairs) >= 5:
        from api.jobs.pnl import spearman_ic
        ic = round(spearman_ic([p[0] for p in ic_pairs], [p[1] for p in ic_pairs]), 4)

    # By regime
    regime_acc: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "hits": 0, "rets": []})
    for r in results:
        reg = r["regime_label"]
        regime_acc[reg]["n"] += 1
        regime_acc[reg]["hits"] += int(r["hit"])
        regime_acc[reg]["rets"].append(r["adj_return"])

    by_regime: Dict[str, Any] = {}
    for reg, acc in sorted(regime_acc.items(), key=lambda x: -x[1]["n"]):
        by_regime[reg] = {
            "n": acc["n"],
            "hit_rate": round(acc["hits"] / acc["n"], 4),
            "avg_return_pct": round(_mean(acc["rets"]) * 100, 3),
        }

    # By signal label
    sig_acc: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "hits": 0, "rets": []})
    for r in results:
        sig_acc[r["signal_label"]]["n"] += 1
        sig_acc[r["signal_label"]]["hits"] += int(r["hit"])
        sig_acc[r["signal_label"]]["rets"].append(r["adj_return"])

    by_signal: Dict[str, Any] = {
        label: {
            "n": acc["n"],
            "hit_rate": round(acc["hits"] / acc["n"], 4),
            "avg_return_pct": round(_mean(acc["rets"]) * 100, 3),
        }
        for label, acc in sig_acc.items()
    }

    equity_curve = {
        d.isoformat() if hasattr(d, "isoformat") else str(d): round(e, 6)
        for d, e in zip(sorted_dates, equity[1:])
    }

    return {
        "total_signals": n,
        "signals_with_return": n,
        "hit_rate": round(hits / n, 4),
        "avg_return_pct": round(_mean(adj_returns) * 100, 4),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "spearman_ic": ic,
        "by_regime": by_regime,
        "by_signal": by_signal,
        "equity_curve": equity_curve,
        "portfolio_returns": [round(r * 100, 4) for r in port_returns],
    }


# ---------------------------------------------------------------------------
# DB data loading
# ---------------------------------------------------------------------------

def load_signals_for_backtest(
    from_date: dt.date,
    to_date: dt.date,
    min_dau: float = 0.0,
    limit: int = _SIGNAL_QUERY_LIMIT,
) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []

    rows = db.safe_fetchall(
        """
        SELECT
            a.symbol,
            a.as_of_date,
            a.regime_label,
            a.deployable_alpha_utility  AS dau,
            COALESCE(psd.signal, 'HOLD') AS signal_label
        FROM axiom_scores_daily a
        LEFT JOIN prosperity_signals_daily psd
            ON psd.symbol = a.symbol
            AND psd.as_of = a.as_of_date
            AND psd.lookback = 252
        WHERE a.as_of_date BETWEEN %s AND %s
          AND COALESCE(a.deployable_alpha_utility, 0) >= %s
        ORDER BY a.as_of_date, a.symbol
        LIMIT %s
        """,
        (from_date, to_date, min_dau, limit),
    )

    return [
        {
            "symbol": r[0],
            "as_of_date": r[1],
            "regime_label": r[2],
            "dau": float(r[3]) if r[3] is not None else None,
            "signal_label": r[4],
        }
        for r in rows
    ]


def load_prices_for_backtest(
    symbols: List[str],
    from_date: dt.date,
    to_date: dt.date,
) -> Dict[str, Dict[Any, float]]:
    if not db.db_read_enabled() or not symbols:
        return {}

    rows = db.safe_fetchall(
        """
        SELECT symbol, date, close
        FROM prosperity_daily_bars
        WHERE symbol = ANY(%s) AND date BETWEEN %s AND %s
        ORDER BY symbol, date
        """,
        (symbols, from_date, to_date),
    )

    price_map: Dict[str, Dict[Any, float]] = {}
    for sym, date, close in rows:
        price_map.setdefault(sym, {})[date] = float(close)
    return price_map


def store_backtest_run(
    run_id: str,
    from_date: dt.date,
    to_date: dt.date,
    horizon_days: int,
    min_dau: float,
    signal_filter: List[str],
    stats: Dict[str, Any],
) -> bool:
    if not db.db_write_enabled():
        return False
    try:
        import json
        db.exec1(
            """
            INSERT INTO axiom_backtest_runs (
                run_id, from_date, to_date, horizon_days, min_dau, signal_filter,
                total_signals, hit_rate, avg_return_pct, sharpe,
                max_drawdown, spearman_ic, result
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                run_id, from_date, to_date, horizon_days, min_dau,
                signal_filter,
                stats.get("total_signals"),
                stats.get("hit_rate"),
                stats.get("avg_return_pct"),
                stats.get("sharpe"),
                stats.get("max_drawdown"),
                stats.get("spearman_ic"),
                json.dumps(stats),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("backtest.store_failed run_id=%s error=%s", run_id, exc)
        return False


def load_recent_runs(limit: int = 10) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT run_id, from_date, to_date, horizon_days, min_dau, signal_filter,
                   total_signals, hit_rate, avg_return_pct, sharpe, max_drawdown,
                   spearman_ic, created_at
            FROM axiom_backtest_runs
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [
            {
                "run_id": r[0],
                "from_date": r[1].isoformat() if hasattr(r[1], "isoformat") else str(r[1]),
                "to_date": r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2]),
                "horizon_days": r[3],
                "min_dau": float(r[4]) if r[4] is not None else None,
                "signal_filter": list(r[5]) if r[5] else [],
                "total_signals": r[6],
                "hit_rate": float(r[7]) if r[7] is not None else None,
                "avg_return_pct": float(r[8]) if r[8] is not None else None,
                "sharpe": float(r[9]) if r[9] is not None else None,
                "max_drawdown": float(r[10]) if r[10] is not None else None,
                "spearman_ic": float(r[11]) if r[11] is not None else None,
                "created_at": r[12].isoformat() if hasattr(r[12], "isoformat") else str(r[12]),
            }
            for r in rows
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_axiom_backtest(
    from_date: dt.date,
    to_date: dt.date,
    *,
    horizon_days: int = 21,
    min_dau: float = 0.0,
    signal_filter: Optional[List[str]] = None,
    store: bool = True,
) -> Dict[str, Any]:
    signal_filter = signal_filter or ["BUY", "SELL"]

    raw_signals = load_signals_for_backtest(from_date, to_date, min_dau)

    symbols = list({s["symbol"] for s in raw_signals})
    price_end = to_date + dt.timedelta(days=horizon_days + 10)
    price_map = load_prices_for_backtest(symbols, from_date, price_end)

    stats = compute_backtest_stats(raw_signals, price_map, horizon_days, signal_filter)

    run_id = str(uuid.uuid4())
    stored = False
    if store and stats.get("total_signals", 0) > 0:
        stored = store_backtest_run(
            run_id, from_date, to_date, horizon_days, min_dau, signal_filter, stats
        )

    logger.info(
        "backtest.run run_id=%s from=%s to=%s horizon=%d signals=%d hit_rate=%s sharpe=%s",
        run_id, from_date, to_date, horizon_days,
        stats.get("total_signals", 0),
        stats.get("hit_rate"), stats.get("sharpe"),
    )

    return {
        "status": "ok",
        "run_id": run_id,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "horizon_days": horizon_days,
        "min_dau": min_dau,
        "signal_filter": signal_filter,
        "stored": stored,
        **stats,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class BacktestRunRequest(BaseModel):
    from_date: str
    to_date: str
    horizon_days: int = Field(default=21, ge=1, le=252)
    min_dau: float = Field(default=0.0, ge=0.0, le=100.0)
    signal_filter: List[str] = Field(default_factory=lambda: ["BUY", "SELL"])
    store: bool = True


def _parse_date(raw: str) -> dt.date:
    return dt.date.fromisoformat(raw)


@router.post("/axiom-backtest/run")
def backtest_run(req: BacktestRunRequest) -> Dict[str, Any]:
    try:
        from_date = _parse_date(req.from_date)
        to_date = _parse_date(req.to_date)
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=f"Invalid date: {exc}") from exc

    if from_date >= to_date:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="from_date must be before to_date")

    signal_filter = [s.upper() for s in req.signal_filter if s.strip()]
    if not signal_filter:
        signal_filter = ["BUY", "SELL"]

    return run_axiom_backtest(
        from_date,
        to_date,
        horizon_days=req.horizon_days,
        min_dau=req.min_dau,
        signal_filter=signal_filter,
        store=req.store,
    )


@router.get("/axiom-backtest/runs")
def backtest_runs(limit: int = Query(default=10, ge=1, le=50)) -> Dict[str, Any]:
    runs = load_recent_runs(limit)
    return {"status": "ok", "count": len(runs), "runs": runs}
