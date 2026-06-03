"""Phase 15.3: SMB Working Capital Optimization Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp


SECTOR_WC_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "Technology": {"ar_days": 45.0, "ap_days": 30.0, "inventory_days": 0.0, "ccc": 15.0},
    "Manufacturing": {"ar_days": 45.0, "ap_days": 45.0, "inventory_days": 30.0, "ccc": 30.0},
    "Retail": {"ar_days": 5.0, "ap_days": 30.0, "inventory_days": 45.0, "ccc": 20.0},
    "Restaurant": {"ar_days": 3.0, "ap_days": 15.0, "inventory_days": 7.0, "ccc": -5.0},
    "Professional Services": {"ar_days": 45.0, "ap_days": 30.0, "inventory_days": 0.0, "ccc": 15.0},
    "Healthcare": {"ar_days": 60.0, "ap_days": 45.0, "inventory_days": 15.0, "ccc": 30.0},
    "Construction": {"ar_days": 60.0, "ap_days": 45.0, "inventory_days": 30.0, "ccc": 45.0},
    "Unknown": {"ar_days": 45.0, "ap_days": 30.0, "inventory_days": 30.0, "ccc": 45.0},
}


@dataclass
class WorkingCapitalAnalysis:
    entity_id: str
    current_cash_conversion_cycle: float
    optimal_cash_conversion_cycle: float
    working_capital_gap_days: float
    trapped_working_capital_usd: float
    ar_days: float
    ap_days: float
    inventory_days: float
    recommendations: List[Dict[str, Any]]
    potential_cash_release_usd: float


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_cash_conversion_cycle(financials: Dict[str, Any]) -> Dict[str, float]:
    """CCC = AR_days + Inventory_days - AP_days."""
    revenue = float(financials.get("revenue") or 1.0)
    cogs = float(financials.get("cogs") or revenue * 0.60)
    ar = float(financials.get("accounts_receivable") or 0.0)
    ap = float(financials.get("accounts_payable") or 0.0)
    inventory = float(financials.get("inventory") or 0.0)

    ar_days = (ar / revenue * 365.0) if revenue > 0 else 0.0
    inventory_days = (inventory / cogs * 365.0) if cogs > 0 else 0.0
    ap_days = (ap / cogs * 365.0) if cogs > 0 else 0.0
    ccc = ar_days + inventory_days - ap_days

    return {
        "ar_days": round(ar_days, 2),
        "ap_days": round(ap_days, 2),
        "inventory_days": round(inventory_days, 2),
        "ccc": round(ccc, 2),
    }


def compute_optimal_ccc(sector: str) -> Dict[str, float]:
    return SECTOR_WC_BENCHMARKS.get(sector, SECTOR_WC_BENCHMARKS["Unknown"])


def compute_trapped_working_capital(
    current_ccc: float,
    optimal_ccc: float,
    annual_revenue: float,
) -> float:
    """Cash trapped above optimal CCC level."""
    gap = current_ccc - optimal_ccc
    if gap <= 0:
        return 0.0
    daily_revenue = annual_revenue / 365.0
    return round(max(0.0, gap * daily_revenue), 2)


def generate_wc_recommendations(
    analysis: WorkingCapitalAnalysis,
    financials: Dict[str, Any],
    sector: str = "Unknown",
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    benchmark = compute_optimal_ccc(sector)
    annual_revenue = float(financials.get("revenue") or 0.0) * 12
    annual_cogs = float(financials.get("cogs") or 0.0) * 12

    # AR recommendation
    if analysis.ar_days > benchmark["ar_days"] + 10:
        excess_days = analysis.ar_days - benchmark["ar_days"]
        estimated_release = round(max(0.0, excess_days / 365.0 * annual_revenue), 2)
        recs.append({
            "action": "accelerate_collections",
            "description": "Implement electronic invoicing and early payment discounts (2/10 net 30)",
            "estimated_cash_release_usd": estimated_release,
            "complexity": "low",
            "timeframe": "30-60 days",
        })
        recs.append({
            "action": "invoice_factoring",
            "description": "Consider invoice factoring for immediate cash release on outstanding AR",
            "estimated_cash_release_usd": round(estimated_release * 0.85, 2),
            "complexity": "medium",
            "timeframe": "1-2 weeks",
        })

    # AP recommendation
    if analysis.ap_days < benchmark["ap_days"] - 10:
        gap_days = benchmark["ap_days"] - analysis.ap_days
        estimated_release = round(max(0.0, gap_days / 365.0 * annual_cogs), 2)
        recs.append({
            "action": "extend_payables",
            "description": "Negotiate extended payment terms with top suppliers (net 45-60)",
            "estimated_cash_release_usd": estimated_release,
            "complexity": "medium",
            "timeframe": "60-90 days",
        })

    # Inventory recommendation
    bench_inv = benchmark.get("inventory_days", 0.0)
    if bench_inv > 0 and analysis.inventory_days > bench_inv + 15:
        excess_inv = analysis.inventory_days - bench_inv
        estimated_release = round(max(0.0, excess_inv / 365.0 * annual_cogs), 2)
        recs.append({
            "action": "reduce_inventory",
            "description": "Implement just-in-time inventory management and reduce safety stock",
            "estimated_cash_release_usd": estimated_release,
            "complexity": "high",
            "timeframe": "90-180 days",
        })
        recs.append({
            "action": "liquidate_slow_moving",
            "description": "Identify slow-moving SKUs for liquidation or return",
            "estimated_cash_release_usd": round(estimated_release * 0.40, 2),
            "complexity": "low",
            "timeframe": "30-60 days",
        })

    return recs


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def build_working_capital_analysis(
    entity_id: str,
    financials: Dict[str, Any],
    sector: str = "Unknown",
) -> WorkingCapitalAnalysis:
    ccc_data = compute_cash_conversion_cycle(financials)
    benchmark = compute_optimal_ccc(sector)
    annual_revenue = float(financials.get("revenue") or 0.0) * 12

    gap_days = ccc_data["ccc"] - benchmark["ccc"]
    trapped = compute_trapped_working_capital(ccc_data["ccc"], benchmark["ccc"], annual_revenue)

    analysis = WorkingCapitalAnalysis(
        entity_id=entity_id,
        current_cash_conversion_cycle=ccc_data["ccc"],
        optimal_cash_conversion_cycle=benchmark["ccc"],
        working_capital_gap_days=round(gap_days, 2),
        trapped_working_capital_usd=trapped,
        ar_days=ccc_data["ar_days"],
        ap_days=ccc_data["ap_days"],
        inventory_days=ccc_data["inventory_days"],
        recommendations=[],
        potential_cash_release_usd=trapped,
    )
    analysis.recommendations = generate_wc_recommendations(analysis, financials, sector)
    total_release = sum(r["estimated_cash_release_usd"] for r in analysis.recommendations)
    analysis.potential_cash_release_usd = round(max(trapped, total_release), 2)
    return analysis
