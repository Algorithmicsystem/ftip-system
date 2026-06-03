"""Phase 14.6: Family Office and RIA Intelligence API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/family-office",
    tags=["family_office"],
    dependencies=[Depends(require_tier("enterprise"))],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PositionIn(BaseModel):
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
    metadata: Dict[str, Any] = {}


class SnapshotIn(BaseModel):
    portfolio_id: str
    family_office_name: str
    positions: List[PositionIn]
    as_of_date: Optional[str] = None


class GoalIn(BaseModel):
    goal_id: str
    goal_type: str
    label: str
    target_amount_usd: float
    target_date_years: float
    current_funding_usd: float
    required_return_annual: float
    risk_budget: float


class GoalAnalysisIn(BaseModel):
    portfolio_id: str
    positions: List[PositionIn]
    goals: List[GoalIn]
    expected_annual_return: float = 0.07
    axiom_regime: str = "TRENDING"


class LiabilityIn(BaseModel):
    label: str
    amount_usd: float
    years_until_due: float


class LiabilityMatchIn(BaseModel):
    portfolio_id: str
    positions: List[PositionIn]
    liabilities: List[LiabilityIn]


class EstateIn(BaseModel):
    portfolio_value_usd: float
    estate_exemption_usd: float = 13_610_000
    state_exemption_usd: float = 0.0
    family_discount_pct: float = 0.35
    annual_gift_exclusion: float = 18_000
    n_gift_recipients: int = 4
    projected_growth_rate: float = 0.07
    trust_horizon_years: int = 100


class RIAClientIn(BaseModel):
    client_id: str
    client_name: str
    portfolio_value_usd: float
    risk_tolerance: str
    time_horizon_years: float
    income_need_annual: float
    tax_bracket: float
    esg_preference: bool = False
    axiom_score: Optional[float] = None
    last_review_date: Optional[str] = None
    sri_score: Optional[float] = None
    metadata: Dict[str, Any] = {}


class RIABookIn(BaseModel):
    clients: List[RIAClientIn]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positions_from_body(raw: List[PositionIn]):
    from api.family_office.multi_asset import PortfolioPosition
    return [
        PortfolioPosition(
            position_id=p.position_id,
            asset_class=p.asset_class,
            ticker_or_id=p.ticker_or_id,
            weight=p.weight,
            current_value_usd=p.current_value_usd,
            cost_basis_usd=p.cost_basis_usd,
            unrealized_gain_pct=p.unrealized_gain_pct,
            axiom_score=p.axiom_score,
            pe_health_score=p.pe_health_score,
            duration_years=p.duration_years,
            credit_rating=p.credit_rating,
            metadata=p.metadata,
        )
        for p in raw
    ]


def _snapshot_from_positions(portfolio_id: str, family_office_name: str, positions, as_of_date=None):
    from api.family_office.multi_asset import build_portfolio_snapshot
    parsed_date = None
    if as_of_date:
        try:
            parsed_date = dt.date.fromisoformat(as_of_date)
        except ValueError:
            pass
    return build_portfolio_snapshot(portfolio_id, family_office_name, positions, parsed_date)


# ---------------------------------------------------------------------------
# Multi-asset portfolio endpoints
# ---------------------------------------------------------------------------

@router.post("/portfolio/snapshot")
def portfolio_snapshot(body: SnapshotIn):
    positions = _positions_from_body(body.positions)
    snapshot = _snapshot_from_positions(
        body.portfolio_id, body.family_office_name, positions, body.as_of_date
    )
    from api.family_office.multi_asset import compute_portfolio_axiom_score, save_portfolio_snapshot
    axiom_overlay = compute_portfolio_axiom_score(positions, snapshot.as_of_date)
    save_portfolio_snapshot(snapshot)
    return {
        "portfolio_id": snapshot.portfolio_id,
        "family_office_name": snapshot.family_office_name,
        "as_of_date": snapshot.as_of_date.isoformat(),
        "total_value_usd": snapshot.total_value_usd,
        "asset_class_allocation": snapshot.asset_class_allocation,
        "equity_weight": snapshot.equity_weight,
        "fixed_income_weight": snapshot.fixed_income_weight,
        "alternatives_weight": snapshot.alternatives_weight,
        "cash_weight": snapshot.cash_weight,
        "axiom_overlay": axiom_overlay,
    }


@router.get("/asset-class-profiles")
def asset_class_profiles():
    from api.family_office.multi_asset import ASSET_CLASS_PROFILES
    return {"profiles": ASSET_CLASS_PROFILES}


# ---------------------------------------------------------------------------
# Concentration risk endpoints
# ---------------------------------------------------------------------------

@router.post("/portfolio/concentration-risk")
def portfolio_concentration_risk(body: SnapshotIn):
    from api.family_office.concentration_risk import build_concentration_report
    positions = _positions_from_body(body.positions)
    snapshot = _snapshot_from_positions(
        body.portfolio_id, body.family_office_name, positions, body.as_of_date
    )
    report = build_concentration_report(
        body.portfolio_id, positions, snapshot.total_value_usd
    )
    return {
        "portfolio_id": report.portfolio_id,
        "concentration_index": report.concentration_index,
        "concentration_risk_score": report.concentration_risk_score,
        "largest_position": report.largest_position,
        "sector_concentration": report.sector_concentration,
        "single_stock_risk": report.single_stock_risk,
        "diversification_recommendations": report.diversification_recommendations,
        "tax_aware_diversification": report.tax_aware_diversification,
    }


@router.post("/portfolio/tax-aware-diversification")
def tax_aware_diversification(body: SnapshotIn, tax_rate: float = Query(0.238)):
    from api.family_office.concentration_risk import compute_tax_aware_diversification
    positions = _positions_from_body(body.positions)
    return compute_tax_aware_diversification(positions, tax_rate)


# ---------------------------------------------------------------------------
# Goal-based intelligence endpoints
# ---------------------------------------------------------------------------

@router.post("/portfolio/goal-analysis")
def portfolio_goal_analysis(body: GoalAnalysisIn):
    from api.family_office.goal_based import (
        InvestmentGoal,
        compute_goal_funding_status,
        compute_probability_of_success,
        generate_goal_based_recommendations,
    )
    from api.family_office.multi_asset import build_portfolio_snapshot

    positions = _positions_from_body(body.positions)
    snapshot = build_portfolio_snapshot(body.portfolio_id, "", positions)

    goals = [
        InvestmentGoal(
            goal_id=g.goal_id,
            goal_type=g.goal_type,
            label=g.label,
            target_amount_usd=g.target_amount_usd,
            target_date_years=g.target_date_years,
            current_funding_usd=g.current_funding_usd,
            required_return_annual=g.required_return_annual,
            risk_budget=g.risk_budget,
        )
        for g in body.goals
    ]

    funding_statuses = [
        compute_goal_funding_status(goal, snapshot.total_value_usd, body.expected_annual_return)
        for goal in goals
    ]

    probabilities = [
        compute_probability_of_success(
            required_return=goal.required_return_annual,
            expected_return=body.expected_annual_return,
            volatility_annual=0.15,
            years=goal.target_date_years,
        )
        for goal in goals
    ]

    recommendations = generate_goal_based_recommendations(goals, snapshot, body.axiom_regime)

    return {
        "portfolio_id": body.portfolio_id,
        "goal_funding_statuses": funding_statuses,
        "monte_carlo_probabilities": probabilities,
        "recommendations": recommendations,
    }


@router.post("/portfolio/liability-matching")
def portfolio_liability_matching(body: LiabilityMatchIn):
    from api.family_office.goal_based import compute_liability_matching_score
    from api.family_office.multi_asset import build_portfolio_snapshot

    positions = _positions_from_body(body.positions)
    snapshot = build_portfolio_snapshot(body.portfolio_id, "", positions)
    liabilities = [l.dict() for l in body.liabilities]
    return compute_liability_matching_score(snapshot, liabilities)


# ---------------------------------------------------------------------------
# Estate intelligence endpoints
# ---------------------------------------------------------------------------

@router.post("/estate/analysis")
def estate_analysis(body: EstateIn):
    from api.family_office.estate_intelligence import build_estate_intelligence_report
    return build_estate_intelligence_report(
        portfolio_value_usd=body.portfolio_value_usd,
        estate_exemption_usd=body.estate_exemption_usd,
        state_exemption_usd=body.state_exemption_usd,
        family_discount_pct=body.family_discount_pct,
        annual_gift_exclusion=body.annual_gift_exclusion,
        n_gift_recipients=body.n_gift_recipients,
        projected_growth_rate=body.projected_growth_rate,
        trust_horizon_years=body.trust_horizon_years,
    )


@router.post("/estate/gifting")
def estate_gifting(body: EstateIn):
    from api.family_office.estate_intelligence import compute_gifting_intelligence
    return compute_gifting_intelligence(
        portfolio_value_usd=body.portfolio_value_usd,
        annual_gift_exclusion=body.annual_gift_exclusion,
        n_recipients=body.n_gift_recipients,
    )


@router.post("/estate/dynasty-trust")
def estate_dynasty_trust(body: EstateIn):
    from api.family_office.estate_intelligence import compute_dynasty_trust_analysis
    return compute_dynasty_trust_analysis(
        portfolio_value_usd=body.portfolio_value_usd,
        projected_growth_rate=body.projected_growth_rate,
        trust_horizon_years=body.trust_horizon_years,
    )


# ---------------------------------------------------------------------------
# RIA intelligence endpoints
# ---------------------------------------------------------------------------

@router.post("/ria/client-brief")
def ria_client_brief(
    client: RIAClientIn,
    positions: List[PositionIn],
    axiom_context: Dict[str, Any] = {},
):
    from api.family_office.multi_asset import build_portfolio_snapshot
    from api.family_office.ria_intelligence import RIAClientProfile, generate_client_brief
    import datetime as dt

    pos_list = _positions_from_body(positions)
    snapshot = build_portfolio_snapshot(client.client_id, client.client_name, pos_list)

    last_review = None
    if client.last_review_date:
        try:
            last_review = dt.date.fromisoformat(client.last_review_date)
        except ValueError:
            pass

    profile = RIAClientProfile(
        client_id=client.client_id,
        client_name=client.client_name,
        portfolio_value_usd=client.portfolio_value_usd,
        risk_tolerance=client.risk_tolerance,
        time_horizon_years=client.time_horizon_years,
        income_need_annual=client.income_need_annual,
        tax_bracket=client.tax_bracket,
        esg_preference=client.esg_preference,
        axiom_score=client.axiom_score,
        last_review_date=last_review,
        sri_score=client.sri_score,
        metadata=client.metadata,
    )
    return generate_client_brief(profile, snapshot, axiom_context)


@router.post("/ria/review-triggers")
def ria_review_triggers(
    client: RIAClientIn,
    axiom_regime: str = Query("TRENDING"),
    sri_score: float = Query(50.0),
):
    from api.family_office.ria_intelligence import RIAClientProfile, compute_client_review_trigger
    import datetime as dt

    last_review = None
    if client.last_review_date:
        try:
            last_review = dt.date.fromisoformat(client.last_review_date)
        except ValueError:
            pass

    profile = RIAClientProfile(
        client_id=client.client_id,
        client_name=client.client_name,
        portfolio_value_usd=client.portfolio_value_usd,
        risk_tolerance=client.risk_tolerance,
        time_horizon_years=client.time_horizon_years,
        income_need_annual=client.income_need_annual,
        tax_bracket=client.tax_bracket,
        esg_preference=client.esg_preference,
        axiom_score=client.axiom_score,
        last_review_date=last_review,
        sri_score=sri_score,
        metadata=client.metadata,
    )
    return compute_client_review_trigger(profile, axiom_regime, sri_score)


@router.post("/ria/book-analytics")
def ria_book_analytics(body: RIABookIn):
    from api.family_office.ria_intelligence import RIAClientProfile, compute_ria_book_analytics
    import datetime as dt

    profiles = []
    for c in body.clients:
        last_review = None
        if c.last_review_date:
            try:
                last_review = dt.date.fromisoformat(c.last_review_date)
            except ValueError:
                pass
        profiles.append(RIAClientProfile(
            client_id=c.client_id,
            client_name=c.client_name,
            portfolio_value_usd=c.portfolio_value_usd,
            risk_tolerance=c.risk_tolerance,
            time_horizon_years=c.time_horizon_years,
            income_need_annual=c.income_need_annual,
            tax_bracket=c.tax_bracket,
            esg_preference=c.esg_preference,
            axiom_score=c.axiom_score,
            last_review_date=last_review,
            sri_score=c.sri_score,
            metadata=c.metadata,
        ))
    return compute_ria_book_analytics(profiles)
