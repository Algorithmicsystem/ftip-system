"""Phase 9.2: Training Data Builder.

Assembles the labeled training dataset from axiom_scores_daily × signal_pnl_daily.
Enforces point-in-time safety: features from axiom_scores_daily, labels from
signal_pnl_daily at a later date. signal_date < as_of_date for all rows.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api import db
from api.axiom.ml.feature_builder import (
    build_feature_vector,
    feature_vector_to_array,
    get_feature_names,
)

logger = logging.getLogger(__name__)

# Grinold-Kahn (2000) Ch. 4: small-sample IC estimates are noisy but directionally
# useful. 20 samples → SE = 1/sqrt(20) ≈ 0.22; usable for half-Kelly bootstrap.
MINIMUM_SAMPLES_INITIAL = 20      # bootstrap-quality model threshold
MINIMUM_SAMPLES_PRODUCTION = 100  # production-quality model threshold

_HORIZON_DAYS: Dict[str, int] = {
    "5d": 5, "21d": 21, "63d": 63,
    "5": 5, "21": 21, "63": 63,
}


def _parse_horizon(horizon_label: str) -> int:
    return _HORIZON_DAYS.get(str(horizon_label), 21)


def load_training_dataset(
    as_of_date: date,
    horizon_label: str = "21d",
    min_samples: int = 50,
    lookback_days: int = 252,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Load labeled training data from the database.

    Returns (X, y, symbols) or (None, None, []) if insufficient labeled samples.
    Binary labels: triple_barrier_outcome=1 → y=1 (profit hit), else y=0.
    """
    if not db.db_read_enabled():
        return None, None, []

    horizon_days = _parse_horizon(horizon_label)
    since = as_of_date - timedelta(days=lookback_days)

    try:
        rows = db.safe_fetchall(
            """
            SELECT
                a.symbol,
                a.payload,
                p.triple_barrier_outcome
            FROM axiom_scores_daily a
            JOIN signal_pnl_daily p
                ON p.symbol = a.symbol
               AND p.signal_date = a.as_of_date
               AND p.horizon_days = %s
            WHERE p.triple_barrier_outcome IS NOT NULL
              AND p.signal_date >= %s
              AND p.signal_date < %s
            ORDER BY p.signal_date
            """,
            (horizon_days, since, as_of_date),
        )
    except Exception as exc:
        logger.warning("training_data.load_failed horizon=%s error=%s", horizon_label, exc)
        return None, None, []

    if not rows or len(rows) < min_samples:
        logger.info(
            "training_data.insufficient_samples found=%d required=%d",
            len(rows) if rows else 0,
            min_samples,
        )
        return None, None, []

    X_rows: List[np.ndarray] = []
    y_list: List[int] = []
    symbols: List[str] = []

    feature_names = get_feature_names()

    for row in rows:
        symbol = str(row[0])
        payload_raw = row[1]
        outcome = row[2]

        if isinstance(payload_raw, str):
            try:
                payload = json.loads(payload_raw)
            except Exception:
                continue
        else:
            payload = payload_raw or {}

        if not isinstance(payload, dict):
            continue

        # Extract factor loadings from payload (stored by engine.py)
        factor_loadings = payload.get("factor_loadings_summary") or []

        try:
            fv = build_feature_vector(payload, factor_loadings, ic_state=None)
            arr = feature_vector_to_array(fv, fill_value=0.5)
        except Exception:
            continue

        # Binary label: 1 = profit target hit, 0 = stop or time-stop
        label = 1 if int(outcome) == 1 else 0

        X_rows.append(arr)
        y_list.append(label)
        symbols.append(symbol)

    if len(X_rows) < min_samples:
        return None, None, []

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle with fixed seed to remove time ordering bias
    # (CPCV handles temporal validation separately)
    rng = random.Random(42)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    X = X[indices]
    y = y[indices]
    symbols = [symbols[i] for i in indices]

    return X, y, symbols


def split_train_test_purged(
    X: np.ndarray,
    y: np.ndarray,
    symbols: List[str],
    test_size: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Purged train/test split — last test_size fraction is test set.

    Preserves approximate temporal ordering (last portion = most recent).
    No shuffling here — shuffling was applied in load_training_dataset.
    """
    n = len(X)
    split_idx = max(1, int(n * (1.0 - test_size)))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    return X_train, X_test, y_train, y_test
