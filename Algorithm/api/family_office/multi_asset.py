"""Phase 14.1: Multi-Asset Portfolio Intelligence."""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset class profiles
# ---------------------------------------------------------------------------

ASSET_CLASS_PROFILES: Dict[str, Dict[str, Any]] = {
    "equity_us": {
        "primary_engine": "axiom_full",
        "risk_model": "historical_var",
        "factor_model": "12_factor",
        "liquidity": "daily",
        "regime_sensitive": True,
    },
    "equity_international": {
        "primary_engine": "axiom_simplified",
        "risk_model": "historical_var",
        "factor_model": "6_factor",
        "liquidity": "daily",
        "regime_sensitive": True,
    },
    "fixed_income_investment_grade": {
        "primary_engine": "duration_credit",
        "risk_model": "parametric_var",
        "factor_model": "rates_credit",
        "liquidity": "daily",
        "regime_sensitive": True,
    },
    "fixed_income_high_yield": {
        "primary_engine": "duration_credit",
        "risk_model": "historical_var",
        "factor_model": "rates_credit_spread",
        "liquidity": "weekly",
        "regime_sensitive": True,
    },
    "private_equity": {
        "primary_engine": "pe_health",
        "risk_model": "stress_test",
        "factor_model": "pe_factors",
        "liquidity": "quarterly",
        "regime_sensitive": False,
    },
    "real_estate": {
        "primary_engine": "reit_axiom",
        "risk_model": "parametric_var",
        "factor_model": "cap_rate_duration",
        "liquidity": "daily",
        "regime_sensitive": True,
    },
    "commodities": {
        "primary_engine": "commodity_carry",
        "risk_model": "historical_var",
        "factor_model": "commodity_factors",
        "liquidity": "daily",
        "regime_sensitive": True,
    },
    "cash": {
        "primary_engine": "none",
        "risk_model": "none",
        "factor_model": "none",
        "liquidity": "immediate",
        "regime_sensitive": False,
    },
}

_EQUITY_CLASSES = {"equity_us", "equity_international"}
_FIXED_INCOME_CLASSES = {"fixed_income_investment_grade", "fixed_income_high_yield"}
_ALTERNATIVES_CLASSES = {"private_equity", "real_estate", "commodities"}
_RATE_SENSITIVE_REGIMES = {"HIGH_VOL", "compensation_capture"}


@dataclass
class PortfolioPosition:
    position_id: str
    asset_class: str
    ticker_or_id: str
    weight: float
    current_value_usd: float
    cost_basis_usd: float
    unrealized_gain_pct: float
    axiom_score: Optional[float] = None
    pe_health_score: Optional[float] = None
    duration_years: Optional[float] = None
    credit_rating: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    portfolio_id: str
    family_office_name: str
    as_of_date: dt.date
    total_value_usd: float
    positions: List[PortfolioPosition]
    asset_class_allocation: Dict[str, float]
    equity_weight: float
    fixed_income_weight: float
    alternatives_weight: float
    cash_weight: float


# ---------------------------------------------------------------------------
# Allocation computation
# ---------------------------------------------------------------------------

def compute_asset_class_allocation(positions: List[PortfolioPosition]) -> Dict[str, float]:
    """Group positions by asset_class and compute total weight per class."""
    allocation: Dict[str, float] = {}
    for pos in positions:
        allocation[pos.asset_class] = allocation.get(pos.asset_class, 0.0) + pos.weight
    return {k: round(v, 6) for k, v in allocation.items()}


def _bucket_weight(positions: List[PortfolioPosition], classes: set) -> float:
    return sum(p.weight for p in positions if p.asset_class in classes)


def build_portfolio_snapshot(
    portfolio_id: str,
    family_office_name: str,
    positions: List[PortfolioPosition],
    as_of_date: Optional[dt.date] = None,
) -> PortfolioSnapshot:
    today = as_of_date or dt.date.today()
    total = sum(p.current_value_usd for p in positions)
    allocation = compute_asset_class_allocation(positions)
    return PortfolioSnapshot(
        portfolio_id=portfolio_id,
        family_office_name=family_office_name,
        as_of_date=today,
        total_value_usd=total,
        positions=positions,
        asset_class_allocation=allocation,
        equity_weight=round(_bucket_weight(positions, _EQUITY_CLASSES), 6),
        fixed_income_weight=round(_bucket_weight(positions, _FIXED_INCOME_CLASSES), 6),
        alternatives_weight=round(_bucket_weight(positions, _ALTERNATIVES_CLASSES), 6),
        cash_weight=round(_bucket_weight(positions, {"cash"}), 6),
    )


# ---------------------------------------------------------------------------
# Duration risk score
# ---------------------------------------------------------------------------

def compute_duration_risk_score(
    duration_years: float,
    regime_label: str,
    current_10y_yield: float = 4.5,
) -> float:
    """Duration risk score (0–100, higher = lower risk / more defensive)."""
    base = 100.0 - clamp(duration_years / 10.0 * 50.0, 0.0, 50.0)
    regime_penalty = 20.0 if regime_label.upper() in _RATE_SENSITIVE_REGIMES else 0.0
    yield_penalty = clamp((current_10y_yield - 3.0) / 4.0 * 20.0, 0.0, 20.0)
    return round(clamp(base - regime_penalty - yield_penalty, 0.0, 100.0), 2)


# ---------------------------------------------------------------------------
# Portfolio AXIOM score overlay
# ---------------------------------------------------------------------------

def compute_portfolio_axiom_score(
    positions: List[PortfolioPosition],
    as_of_date: dt.date,
) -> Dict[str, Any]:
    """Compute weighted AXIOM score overlay across all asset classes."""
    if not positions:
        return {
            "weighted_axiom_score": 50.0,
            "equity_axiom_score": 50.0,
            "pe_health_score": 50.0,
            "fixed_income_score": 50.0,
            "score_coverage": 0.0,
        }

    # Pull equity AXIOM scores from DB if not pre-populated
    if db.db_read_enabled():
        for pos in positions:
            if pos.asset_class in _EQUITY_CLASSES and pos.axiom_score is None:
                try:
                    row = db.safe_fetchone(
                        """
                        SELECT payload->>'deployable_alpha_utility'
                          FROM axiom_scores_daily
                         WHERE symbol = %s AND as_of_date = %s
                        """,
                        (pos.ticker_or_id, as_of_date),
                    )
                    if row and row[0] is not None:
                        pos.axiom_score = float(row[0])
                except Exception:
                    pass

    # Pull latest regime for duration scoring
    regime = "TRENDING"
    if db.db_read_enabled():
        try:
            row = db.safe_fetchone(
                "SELECT regime_label FROM market_breadth_daily ORDER BY as_of_date DESC LIMIT 1"
            )
            if row and row[0]:
                regime = str(row[0])
        except Exception:
            pass

    weighted_sum = 0.0
    total_weight = 0.0
    eq_sum = eq_w = 0.0
    pe_sum = pe_w = 0.0
    fi_sum = fi_w = 0.0
    scored_weight = 0.0

    for pos in positions:
        w = pos.weight
        score: Optional[float] = None

        if pos.asset_class in _EQUITY_CLASSES:
            score = pos.axiom_score  # may still be None
            if score is not None:
                eq_sum += score * w
                eq_w += w
        elif pos.asset_class == "private_equity":
            score = pos.pe_health_score
            if score is not None:
                pe_sum += score * w
                pe_w += w
        elif pos.asset_class in _FIXED_INCOME_CLASSES:
            dur = pos.duration_years or 5.0
            score = compute_duration_risk_score(dur, regime)
            fi_sum += score * w
            fi_w += w
        elif pos.asset_class == "cash":
            score = 100.0
        else:
            score = 50.0

        effective_score = score if score is not None else 50.0
        weighted_sum += effective_score * w
        total_weight += w
        if score is not None:
            scored_weight += w

    wt = total_weight or 1.0
    return {
        "weighted_axiom_score": round(clamp(weighted_sum / wt, 0.0, 100.0), 2),
        "equity_axiom_score": round(eq_sum / eq_w, 2) if eq_w > 0 else 50.0,
        "pe_health_score": round(pe_sum / pe_w, 2) if pe_w > 0 else 50.0,
        "fixed_income_score": round(fi_sum / fi_w, 2) if fi_w > 0 else 50.0,
        "score_coverage": round(scored_weight / wt, 4) if wt > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_portfolio_snapshot(snapshot: PortfolioSnapshot) -> bool:
    if not db.db_write_enabled():
        return False
    try:
        positions_json = json.dumps([
            {
                "position_id": p.position_id,
                "asset_class": p.asset_class,
                "ticker_or_id": p.ticker_or_id,
                "weight": p.weight,
                "current_value_usd": p.current_value_usd,
                "cost_basis_usd": p.cost_basis_usd,
                "unrealized_gain_pct": p.unrealized_gain_pct,
                "axiom_score": p.axiom_score,
                "pe_health_score": p.pe_health_score,
                "duration_years": p.duration_years,
                "credit_rating": p.credit_rating,
                "metadata": p.metadata,
            }
            for p in snapshot.positions
        ])
        db.safe_execute(
            """
            INSERT INTO family_office_portfolios
                (portfolio_id, family_office_name, as_of_date, total_value_usd,
                 positions, asset_class_allocation, created_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
            ON CONFLICT (portfolio_id) DO UPDATE SET
                total_value_usd      = EXCLUDED.total_value_usd,
                positions            = EXCLUDED.positions,
                asset_class_allocation = EXCLUDED.asset_class_allocation,
                created_at           = now()
            """,
            (
                snapshot.portfolio_id,
                snapshot.family_office_name,
                snapshot.as_of_date,
                snapshot.total_value_usd,
                positions_json,
                json.dumps(snapshot.asset_class_allocation),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("save_portfolio_snapshot failed portfolio=%s err=%s", snapshot.portfolio_id, exc)
        return False
