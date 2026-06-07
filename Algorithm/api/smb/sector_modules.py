"""Phase 15.5: SMB Sector-Specific Intelligence Modules."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

from api.assistant.phase3.common import clamp


# ===========================================================================
# MODULE A — Restaurant Intelligence
# ===========================================================================

@dataclass
class RestaurantMetrics:
    entity_id: str
    revenue_per_seat: float
    food_cost_pct: float
    labor_cost_pct: float
    prime_cost_pct: float
    table_turn_rate: float
    average_check: float
    restaurant_intelligence_score: float


def compute_restaurant_intelligence(
    financials: Dict[str, Any],
    operating_data: Dict[str, Any],
) -> RestaurantMetrics:
    entity_id = str(financials.get("entity_id") or operating_data.get("entity_id") or "")
    revenue = float(financials.get("revenue") or operating_data.get("revenue") or 0.0)
    seats = float(operating_data.get("seats") or 50.0)
    food_cost_pct = float(operating_data.get("food_cost_pct") or 0.32)
    labor_cost_pct = float(operating_data.get("labor_cost_pct") or 0.33)
    table_turns = float(operating_data.get("table_turn_rate") or 2.0)
    avg_check = float(operating_data.get("average_check") or 0.0)
    covers = float(operating_data.get("covers_per_day") or 0.0)

    prime_cost_pct = food_cost_pct + labor_cost_pct
    revenue_per_seat = (revenue / seats) if seats > 0 else 0.0

    # Prime cost score (weight 0.70)
    if prime_cost_pct < 0.60:
        prime_score = 90.0
    elif prime_cost_pct < 0.65:
        prime_score = 75.0
    elif prime_cost_pct < 0.70:
        prime_score = 50.0
    else:
        prime_score = 20.0

    # Food cost score (weight 0.15)
    if 0.28 <= food_cost_pct <= 0.32:
        food_score = 80.0
    elif 0.25 <= food_cost_pct < 0.28 or 0.32 < food_cost_pct <= 0.38:
        food_score = 60.0
    else:
        food_score = 35.0

    # Revenue per seat score (weight 0.15)
    annual_rps = revenue_per_seat * 12 if revenue < 100_000 else revenue_per_seat
    if annual_rps >= 10_000:
        rps_score = 80.0
    elif annual_rps >= 5_000:
        rps_score = 65.0
    elif annual_rps > 0:
        rps_score = 45.0
    else:
        rps_score = 50.0

    total_score = round(clamp(
        prime_score * 0.70 + food_score * 0.15 + rps_score * 0.15,
        0.0, 100.0,
    ), 2)

    return RestaurantMetrics(
        entity_id=entity_id,
        revenue_per_seat=round(revenue_per_seat, 2),
        food_cost_pct=round(food_cost_pct, 4),
        labor_cost_pct=round(labor_cost_pct, 4),
        prime_cost_pct=round(prime_cost_pct, 4),
        table_turn_rate=table_turns,
        average_check=avg_check,
        restaurant_intelligence_score=total_score,
    )


# ===========================================================================
# MODULE B — Manufacturing Intelligence
# ===========================================================================

@dataclass
class ManufacturingMetrics:
    entity_id: str
    oee_proxy: float
    inventory_turns: float
    gross_margin_vs_benchmark: float
    supply_chain_concentration: float
    manufacturing_intelligence_score: float


def compute_manufacturing_intelligence(
    financials: Dict[str, Any],
    operating_data: Dict[str, Any],
) -> ManufacturingMetrics:
    entity_id = str(financials.get("entity_id") or operating_data.get("entity_id") or "")
    revenue = float(financials.get("revenue") or 1.0)
    cogs = float(financials.get("cogs") or revenue * 0.65)
    inventory = float(financials.get("inventory") or operating_data.get("inventory") or 0.0)
    gross_margin = float(financials.get("gross_margin") or (1.0 - cogs / revenue) if revenue > 0 else 0.35)
    rev_growth = float(financials.get("revenue_growth_yoy") or 0.0)
    inv_growth = float(financials.get("inventory_growth_yoy") or 0.0)
    supplier_conc = float(operating_data.get("top_supplier_concentration") or financials.get("top_supplier_concentration") or 0.40)

    # Inventory turns: COGS / avg_inventory
    if inventory > 0:
        inventory_turns = cogs / inventory
    else:
        inventory_turns = float(operating_data.get("inventory_turns") or 4.0)

    # OEE proxy: revenue growth / (1 + inventory growth) — high ratio = efficient
    oee_proxy = rev_growth / max(0.01, 1.0 + inv_growth) if rev_growth != 0 else 0.50
    oee_proxy = round(clamp(oee_proxy + 0.50, 0.0, 1.0), 4)  # normalize to 0-1

    # Inventory turns score (benchmark 4-8)
    if 4.0 <= inventory_turns <= 8.0:
        inv_score = 80.0
    elif inventory_turns > 8.0:
        inv_score = 90.0
    elif inventory_turns >= 2.0:
        inv_score = 50.0
    else:
        inv_score = 20.0

    # Gross margin vs sector benchmark (35% for manufacturing)
    MFGR_GM_BENCHMARK = 0.35
    gm_delta = gross_margin - MFGR_GM_BENCHMARK
    gm_score = clamp(60.0 + gm_delta * 200.0, 20.0, 95.0)
    gm_vs_benchmark = round(gm_delta, 4)

    # Supply chain concentration score (lower concentration = better)
    sc_score = clamp(80.0 - supplier_conc * 60.0, 20.0, 80.0)

    total_score = round(clamp(
        inv_score * 0.40 + gm_score * 0.35 + sc_score * 0.25,
        0.0, 100.0,
    ), 2)

    return ManufacturingMetrics(
        entity_id=entity_id,
        oee_proxy=round(oee_proxy, 4),
        inventory_turns=round(inventory_turns, 2),
        gross_margin_vs_benchmark=gm_vs_benchmark,
        supply_chain_concentration=round(supplier_conc, 4),
        manufacturing_intelligence_score=total_score,
    )


# ===========================================================================
# MODULE C — Professional Services Intelligence
# ===========================================================================

@dataclass
class ProfServicesMetrics:
    entity_id: str
    revenue_per_employee: float
    billable_rate: float
    utilization_estimate: float
    client_concentration_risk: float
    pipeline_proxy: float
    profservices_intelligence_score: float


def compute_profservices_intelligence(
    financials: Dict[str, Any],
    operating_data: Dict[str, Any],
) -> ProfServicesMetrics:
    entity_id = str(financials.get("entity_id") or operating_data.get("entity_id") or "")
    revenue = float(financials.get("revenue") or operating_data.get("revenue") or 0.0)
    headcount = float(operating_data.get("headcount") or 1.0)
    billable_hours = float(operating_data.get("billable_hours") or 0.0)
    client_conc = float(operating_data.get("client_concentration") or 0.40)
    ar = float(financials.get("accounts_receivable") or 0.0)
    ar_growth = float(financials.get("receivables_growth") or 0.0)

    rev_per_emp = revenue / headcount if headcount > 0 else 0.0

    # Billable rate
    billable_rate = revenue / billable_hours if billable_hours > 0 else 0.0

    # Utilization estimate
    if billable_hours > 0 and headcount > 0:
        utilization_estimate = min(billable_hours / (headcount * 2080.0), 1.0)
    else:
        utilization_estimate = 0.55  # neutral default

    # Pipeline proxy: declining AR growth + growing revenue = healthy pipeline
    pipeline_proxy = clamp(0.50 - ar_growth * 2.0, 0.20, 0.90)

    # Revenue per employee score (weight 0.65)
    if rev_per_emp >= 150_000:
        rpe_score = 90.0
    elif rev_per_emp >= 100_000:
        rpe_score = 75.0
    elif rev_per_emp >= 75_000:
        rpe_score = 50.0
    elif rev_per_emp >= 50_000:
        rpe_score = 30.0
    else:
        rpe_score = 15.0

    # Utilization score (weight 0.20)
    if utilization_estimate >= 0.75:
        util_score = 90.0
    elif utilization_estimate >= 0.60:
        util_score = 70.0
    elif utilization_estimate >= 0.50:
        util_score = 50.0
    else:
        util_score = 30.0

    # Client concentration penalty (weight 0.15)
    cc_score = clamp(80.0 - client_conc * 80.0, 20.0, 80.0)

    total_score = round(clamp(
        rpe_score * 0.65 + util_score * 0.20 + cc_score * 0.15,
        0.0, 100.0,
    ), 2)

    return ProfServicesMetrics(
        entity_id=entity_id,
        revenue_per_employee=round(rev_per_emp, 2),
        billable_rate=round(billable_rate, 2),
        utilization_estimate=round(utilization_estimate, 4),
        client_concentration_risk=round(client_conc, 4),
        pipeline_proxy=round(pipeline_proxy, 4),
        profservices_intelligence_score=total_score,
    )


# ===========================================================================
# Sector router
# ===========================================================================

_SECTOR_MAP = {
    "Restaurant": compute_restaurant_intelligence,
    "Manufacturing": compute_manufacturing_intelligence,
    "Professional Services": compute_profservices_intelligence,
    "Consulting": compute_profservices_intelligence,
}


RESTAURANT_PRIME_COST_TARGETS = {
    "fast_casual": 0.58,
    "full_service": 0.63,
    "fine_dining": 0.67,
    "fast_food": 0.52,
    "bar": 0.55,
    "default": 0.63,
}


def _enrich_restaurant(metrics_dict: Dict, financials: Dict, operating_data: Dict) -> Dict:
    """Add prime cost benchmarking and action recommendation."""
    restaurant_type = str(operating_data.get("restaurant_type") or "full_service")
    target = RESTAURANT_PRIME_COST_TARGETS.get(restaurant_type, RESTAURANT_PRIME_COST_TARGETS["default"])
    prime_cost = metrics_dict["prime_cost_pct"]
    gap = round(prime_cost - target, 4)

    if prime_cost < target * 0.95:
        status = "excellent"
    elif prime_cost <= target * 1.02:
        status = "on-target"
    elif prime_cost <= target * 1.08:
        status = "watch"
    else:
        status = "critical"

    annual_revenue = float(financials.get("revenue") or 0.0) * 12
    dollar_impact = round(max(0.0, gap * annual_revenue), 2)

    if status == "critical":
        action = f"Prime cost at {prime_cost*100:.1f}% vs {target*100:.1f}% target — reduce labor by scheduling optimization and food cost by menu engineering."
    elif status == "watch":
        action = f"Prime cost slightly elevated ({prime_cost*100:.1f}%). Review food waste and part-time scheduling flexibility."
    elif status == "excellent":
        action = f"Prime cost well-controlled at {prime_cost*100:.1f}%. Maintain current discipline."
    else:
        action = f"Prime cost on target at {prime_cost*100:.1f}%. Minor optimization opportunities available."

    metrics_dict.update({
        "prime_cost_target": target,
        "prime_cost_gap": gap,
        "prime_cost_status": status,
        "prime_cost_dollar_impact": dollar_impact,
        "action": action,
    })
    return metrics_dict


MANUFACTURING_BENCHMARK_UTILIZATION = 0.85


def _enrich_manufacturing(metrics_dict: Dict, financials: Dict, operating_data: Dict) -> Dict:
    """Add capacity utilization and backlog metrics."""
    oee = metrics_dict["oee_proxy"]
    capacity_target = MANUFACTURING_BENCHMARK_UTILIZATION
    underutil_gap = max(0.0, capacity_target - oee)
    annual_revenue = float(financials.get("revenue") or 0.0) * 12
    underutil_cost = round(underutil_gap * annual_revenue, 2)

    backlog = float(operating_data.get("backlog_revenue") or 0.0)
    monthly_rev = annual_revenue / 12 if annual_revenue > 0 else 1.0
    backlog_months = round(backlog / monthly_rev, 2) if monthly_rev > 0 else 0.0

    if backlog_months >= 3:
        backlog_health = "strong"
    elif backlog_months >= 1.5:
        backlog_health = "adequate"
    elif backlog_months >= 0.5:
        backlog_health = "thin"
    else:
        backlog_health = "empty"

    if oee < 0.70:
        action = f"OEE proxy at {oee*100:.0f}% — well below {capacity_target*100:.0f}% target. Annual underutilization cost ~${underutil_cost:,.0f}. Prioritize demand generation or capacity reduction."
    elif oee < capacity_target:
        action = f"OEE proxy at {oee*100:.0f}% — approaching target. Focus on reducing downtime and setup times."
    else:
        action = f"Strong utilization at {oee*100:.0f}%. Consider capacity expansion if backlog exceeds 3 months."

    metrics_dict.update({
        "capacity_utilization_estimate": round(oee, 4),
        "capacity_utilization_target": capacity_target,
        "underutilization_cost_usd": underutil_cost,
        "order_backlog_months": backlog_months,
        "backlog_health": backlog_health,
        "action": action,
    })
    return metrics_dict


PROF_SERVICES_BENCHMARKS = {
    "Professional Services": 120_000,
    "Consulting": 150_000,
    "default": 100_000,
}


def _enrich_profservices(metrics_dict: Dict, financials: Dict, operating_data: Dict) -> Dict:
    """Add sector benchmark and recurring revenue estimate."""
    sector = str(operating_data.get("sector") or "Professional Services")
    benchmark_rpe = PROF_SERVICES_BENCHMARKS.get(sector, PROF_SERVICES_BENCHMARKS["default"])
    rpe = metrics_dict["revenue_per_employee"]

    ar_growth = float(financials.get("receivables_growth") or 0.0)
    recurring_pct = max(0.0, min(1.0, 0.50 - ar_growth))  # declining AR growth = more recurring

    cc_risk = metrics_dict["client_concentration_risk"]
    if cc_risk > 0.60:
        cc_label = "critical"
    elif cc_risk > 0.40:
        cc_label = "elevated"
    elif cc_risk > 0.25:
        cc_label = "moderate"
    else:
        cc_label = "low"

    if rpe >= benchmark_rpe * 1.2:
        action = f"Revenue per employee ${rpe:,.0f} exceeds benchmark ${benchmark_rpe:,.0f}. Focus on retention and upsell."
    elif rpe >= benchmark_rpe * 0.9:
        action = f"Revenue per employee ${rpe:,.0f} near benchmark. Improve utilization rate to expand margin."
    else:
        action = f"Revenue per employee ${rpe:,.0f} below benchmark ${benchmark_rpe:,.0f}. Raise billing rates or increase headcount utilization."

    metrics_dict.update({
        "sector_benchmark_revenue_per_employee": benchmark_rpe,
        "utilization_proxy": round(metrics_dict["utilization_estimate"], 4),
        "recurring_revenue_estimate_pct": round(recurring_pct, 4),
        "client_concentration_risk_label": cc_label,
        "action": action,
    })
    return metrics_dict


def get_sector_module(
    sector: str,
    financials: Dict[str, Any],
    operating_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fn = _SECTOR_MAP.get(sector)
    if fn is None:
        return {"status": "no_module"}
    metrics = fn(financials, operating_data or {})
    result = {"status": "ok", "sector": sector, **dataclasses.asdict(metrics)}
    if sector == "Restaurant":
        result = _enrich_restaurant(result, financials, operating_data or {})
    elif sector == "Manufacturing":
        result = _enrich_manufacturing(result, financials, operating_data or {})
    elif sector in ("Professional Services", "Consulting"):
        result = _enrich_profservices(result, financials, operating_data or {})
    return result
