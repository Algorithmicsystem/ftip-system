from __future__ import annotations

import datetime as dt
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from api.main import Candle

from . import audit as audit_mod
from .ensemble import combine
from .hashing import hash_payload
from .regime import classify_regime
from .strategies import run_strategies


class StrategyGraphResult(Dict[str, object]):
    pass


def _filter_candles(candles: List["Candle"], as_of_date: dt.date) -> List["Candle"]:
    out: List[Candle] = []
    for c in candles:
        try:
            d = dt.date.fromisoformat(c.timestamp)
        except Exception:
            continue
        if d <= as_of_date:
            out.append(c)
    return out


def compute_strategy_graph(
    symbol: str,
    as_of_date: dt.date,
    lookback: int,
    candles_all: List["Candle"],
    *,
    audit_no_lookahead: bool = True,
) -> StrategyGraphResult:
    from api.main import compute_features

    candles = _filter_candles(candles_all, as_of_date)
    if len(candles) < max(10, min(lookback, 30)):
        raise ValueError("insufficient candles for requested lookback")
    window = candles[-min(lookback, len(candles)) :]

    features = compute_features(window)
    regime, regime_meta = classify_regime(window, features)
    strategies = run_strategies(features)
    ensemble = combine(regime, strategies)
    audit = audit_mod.build_audit(window, as_of_date, audit_no_lookahead)

    strategies_hash = hash_payload({"strategies": strategies})
    ensemble_hash = hash_payload({"ensemble": ensemble})

    return StrategyGraphResult(
        symbol=symbol,
        as_of_date=as_of_date.isoformat(),
        lookback=int(lookback),
        regime=regime,
        strategies=strategies,
        ensemble=ensemble,
        audit=audit,
        hashes={"strategies_hash": strategies_hash, "ensemble_hash": ensemble_hash},
        regime_meta=regime_meta,
    )


__all__ = ["compute_strategy_graph", "StrategyGraphResult"]
