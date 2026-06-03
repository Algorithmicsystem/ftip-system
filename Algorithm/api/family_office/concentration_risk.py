"""Phase 14.2: Concentration Risk Intelligence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp
from api.family_office.multi_asset import PortfolioPosition


@dataclass
class ConcentrationRiskReport:
    portfolio_id: str
    largest_position: Dict[str, Any]
    concentration_index: float          # HHI
    concentration_risk_score: float     # HHI × 100
    sector_concentration: Dict[str, float]
    single_stock_risk: Dict[str, Any]
    correlation_to_largest: Dict[str, float]
    diversification_recommendations: List[Dict]
    tax_aware_diversification: Dict[str, Any]


# ---------------------------------------------------------------------------
# HHI
# ---------------------------------------------------------------------------

def compute_concentration_index(positions: List[PortfolioPosition]) -> float:
    """Herfindahl-Hirschman Index = sum of squared weights. Range [1/N, 1.0]."""
    if not positions:
        return 0.0
    return round(sum(p.weight ** 2 for p in positions), 6)


# ---------------------------------------------------------------------------
# Single-stock risk
# ---------------------------------------------------------------------------

def compute_single_stock_risk(
    symbol: str,
    position_weight: float,
    axiom_payload: Dict,
    portfolio_value_usd: float,
) -> Dict[str, Any]:
    """Quantify risk from a concentrated equity position."""
    engines = axiom_payload.get("engine_scores", {})
    frag_engine = engines.get("critical_fragility", {})
    frag_comps = frag_engine.get("components", {})

    fragility = float(
        frag_engine.get("score")
        or frag_comps.get("mtrs_component")
        or 50.0
    )
    mtrs = float(frag_comps.get("mtrs_component") or fragility)
    scps = float(frag_comps.get("scps_component") or 50.0)

    max_drawdown_estimate = (fragility / 100.0) * 0.50
    dollar_at_risk = portfolio_value_usd * position_weight * max_drawdown_estimate

    # Portfolio loss fraction
    portfolio_loss_pct = position_weight * max_drawdown_estimate
    if portfolio_loss_pct >= 0.15:
        risk_label = "extreme"
    elif portfolio_loss_pct >= 0.08:
        risk_label = "significant"
    elif portfolio_loss_pct >= 0.04:
        risk_label = "elevated"
    else:
        risk_label = "manageable"

    # Correlation benefit: how much risk is diversifiable
    # High concentration → low diversification benefit
    correlation_benefit = round(max(0.0, 1.0 - position_weight * 3.0), 4)

    return {
        "symbol": symbol,
        "position_weight": round(position_weight, 4),
        "dollar_at_risk": round(dollar_at_risk, 2),
        "fragility_score": round(fragility, 2),
        "tail_risk_score": round(mtrs, 2),
        "bubble_risk_score": round(scps, 2),
        "max_drawdown_estimate": round(max_drawdown_estimate, 4),
        "correlation_benefit": correlation_benefit,
        "risk_label": risk_label,
    }


# ---------------------------------------------------------------------------
# Diversification recommendations
# ---------------------------------------------------------------------------

def generate_diversification_recommendations(
    positions: List[PortfolioPosition],
    concentration_report: "ConcentrationRiskReport",
    tax_lots: Optional[Dict] = None,
) -> List[Dict]:
    recs: List[Dict] = []
    largest = concentration_report.largest_position
    hhi = concentration_report.concentration_index
    sector_conc = concentration_report.sector_concentration

    lp_weight = largest.get("weight", 0.0)
    lp_ticker = largest.get("ticker", "")
    lp_axiom = largest.get("axiom_score") or 50.0
    lp_fragility = concentration_report.single_stock_risk.get("fragility_score", 50.0)

    # Rule 1: large + fragile → reduce
    if lp_weight > 0.25 and lp_fragility > 60:
        reduction = round((lp_weight - 0.15) * hhi * 50, 1)
        recs.append({
            "action": "reduce",
            "symbol": lp_ticker,
            "rationale": (
                f"{lp_ticker} represents {lp_weight*100:.1f}% of portfolio with fragility "
                f"score {lp_fragility:.0f} — recommend trimming to ≤15%"
            ),
            "priority": "high",
            "estimated_risk_reduction": round(clamp(reduction, 5.0, 40.0), 1),
        })

    # Rule 2: large + high conviction → collar hedge
    if lp_weight > 0.25 and lp_axiom > 70 and lp_fragility <= 60:
        recs.append({
            "action": "hedge",
            "symbol": lp_ticker,
            "rationale": (
                f"{lp_ticker} has strong AXIOM score ({lp_axiom:.0f}) but high concentration "
                "({:.1f}%) — consider collar strategy to preserve upside while limiting downside".format(lp_weight * 100)
            ),
            "priority": "medium",
            "estimated_risk_reduction": 15.0,
        })

    # Rule 3: sector concentration
    for sector, weight in sector_conc.items():
        if weight > 0.40:
            recs.append({
                "action": "add_diversifier",
                "symbol": sector,
                "rationale": (
                    f"{sector} concentration at {weight*100:.1f}% exceeds 40% — "
                    "add uncorrelated sector exposure"
                ),
                "priority": "medium",
                "estimated_risk_reduction": 10.0,
            })

    # Rule 4: HHI-based systematic rebalancing
    if hhi > 0.20:
        recs.append({
            "action": "reduce",
            "symbol": "portfolio_rebalance",
            "rationale": (
                f"Portfolio HHI of {hhi:.3f} exceeds 0.20 threshold — "
                "systematic rebalancing toward equal-weight reduces concentration risk"
            ),
            "priority": "low",
            "estimated_risk_reduction": round(clamp((hhi - 0.20) * 100, 2.0, 30.0), 1),
        })

    return recs


# ---------------------------------------------------------------------------
# Tax-aware diversification
# ---------------------------------------------------------------------------

def compute_tax_aware_diversification(
    positions: List[PortfolioPosition],
    tax_rate: float = 0.238,
) -> Dict[str, Any]:
    """Sequence diversification by tax cost."""
    loss_harvest: List[Dict] = []
    low_cost: List[Dict] = []
    high_cost: List[Dict] = []
    total_tax_cost = 0.0

    for pos in positions:
        gain_pct = pos.unrealized_gain_pct
        gain_usd = pos.current_value_usd * gain_pct
        tax_cost = max(0.0, gain_usd * tax_rate)

        entry = {
            "ticker_or_id": pos.ticker_or_id,
            "unrealized_gain_pct": round(gain_pct, 4),
            "unrealized_gain_usd": round(gain_usd, 2),
            "tax_cost_to_sell_usd": round(tax_cost, 2),
            "weight": round(pos.weight, 4),
        }

        if gain_pct < 0:
            loss_harvest.append(entry)
        elif gain_pct <= 0.15:
            low_cost.append(entry)
            total_tax_cost += tax_cost
        else:
            high_cost.append(entry)
            total_tax_cost += tax_cost

    # Sort high-cost by gain desc (highest tax burden last / most complex)
    high_cost.sort(key=lambda x: x["unrealized_gain_pct"], reverse=True)
    low_cost.sort(key=lambda x: x["tax_cost_to_sell_usd"])

    sequence = (
        [{"step": 1, "action": "harvest_losses", "candidates": [p["ticker_or_id"] for p in loss_harvest]}]
        + [{"step": 2, "action": "diversify_low_cost", "candidates": [p["ticker_or_id"] for p in low_cost]}]
        + [{"step": 3, "action": "use_options_or_gifting", "candidates": [p["ticker_or_id"] for p in high_cost]}]
    )

    return {
        "loss_harvest_candidates": loss_harvest,
        "low_cost_diversification": low_cost,
        "high_cost_positions": high_cost,
        "total_tax_cost_to_full_diversify": round(total_tax_cost, 2),
        "recommended_sequence": sequence,
    }


# ---------------------------------------------------------------------------
# Full concentration report
# ---------------------------------------------------------------------------

def build_concentration_report(
    portfolio_id: str,
    positions: List[PortfolioPosition],
    portfolio_value_usd: float,
    tax_lots: Optional[Dict] = None,
) -> ConcentrationRiskReport:
    if not positions:
        return ConcentrationRiskReport(
            portfolio_id=portfolio_id,
            largest_position={},
            concentration_index=0.0,
            concentration_risk_score=0.0,
            sector_concentration={},
            single_stock_risk={},
            correlation_to_largest={},
            diversification_recommendations=[],
            tax_aware_diversification={},
        )

    largest_pos = max(positions, key=lambda p: p.weight)
    largest_dict = {
        "ticker": largest_pos.ticker_or_id,
        "asset_class": largest_pos.asset_class,
        "weight": round(largest_pos.weight, 4),
        "axiom_score": largest_pos.axiom_score,
    }

    hhi = compute_concentration_index(positions)
    conc_score = round(hhi * 100.0, 2)

    # Sector concentration (from metadata)
    sector_conc: Dict[str, float] = {}
    for pos in positions:
        sector = str(pos.metadata.get("sector") or pos.asset_class)
        sector_conc[sector] = round(sector_conc.get(sector, 0.0) + pos.weight, 4)

    # Single stock risk for largest equity position
    single_risk: Dict = {}
    lp_axiom_payload = largest_pos.metadata.get("axiom_payload") or {}
    if lp_axiom_payload or largest_pos.axiom_score is not None:
        if not lp_axiom_payload:
            lp_axiom_payload = {
                "engine_scores": {
                    "critical_fragility": {
                        "score": 100.0 - (largest_pos.axiom_score or 50.0),
                        "components": {"mtrs_component": 50.0, "scps_component": 50.0},
                    }
                }
            }
        single_risk = compute_single_stock_risk(
            largest_pos.ticker_or_id, largest_pos.weight, lp_axiom_payload, portfolio_value_usd
        )

    tax = compute_tax_aware_diversification(positions)

    report = ConcentrationRiskReport(
        portfolio_id=portfolio_id,
        largest_position=largest_dict,
        concentration_index=hhi,
        concentration_risk_score=conc_score,
        sector_concentration=sector_conc,
        single_stock_risk=single_risk,
        correlation_to_largest={},
        diversification_recommendations=[],
        tax_aware_diversification=tax,
    )
    report.diversification_recommendations = generate_diversification_recommendations(
        positions, report, tax_lots
    )
    return report
