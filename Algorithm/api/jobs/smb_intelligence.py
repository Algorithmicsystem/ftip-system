"""Phase 4: SMB Intelligence Module.

Provides:
  store_smb_financials      — upsert monthly financials for an SMB
  forecast_cash_flow        — 12-month cash flow forecast via linear trend
  compute_supplier_risks    — supplier concentration + cash runway analysis
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _add_months(d: dt.date, n: int) -> dt.date:
    """Return d shifted forward by n months (approximate)."""
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    import calendar
    day = min(d.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_smb_financials(
    entity_id: str,
    month_end: dt.date,
    financials: Dict[str, Any],
) -> bool:
    """Upsert one month of financials for an SMB entity."""
    if not db.db_write_enabled():
        return False
    try:
        db.safe_execute(
            """
            INSERT INTO smb_monthly_financials
                (entity_id, month_end, revenue, cogs, operating_expenses,
                 net_income, cash_balance, accounts_receivable, accounts_payable,
                 inventory, payroll, top_supplier_concentration)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, month_end)
            DO UPDATE SET
                revenue                    = EXCLUDED.revenue,
                cogs                       = EXCLUDED.cogs,
                operating_expenses         = EXCLUDED.operating_expenses,
                net_income                 = EXCLUDED.net_income,
                cash_balance               = EXCLUDED.cash_balance,
                accounts_receivable        = EXCLUDED.accounts_receivable,
                accounts_payable           = EXCLUDED.accounts_payable,
                inventory                  = EXCLUDED.inventory,
                payroll                    = EXCLUDED.payroll,
                top_supplier_concentration = EXCLUDED.top_supplier_concentration,
                reported_at                = now()
            """,
            (
                entity_id,
                month_end,
                financials.get("revenue"),
                financials.get("cogs"),
                financials.get("operating_expenses"),
                financials.get("net_income"),
                financials.get("cash_balance"),
                financials.get("accounts_receivable"),
                financials.get("accounts_payable"),
                financials.get("inventory"),
                financials.get("payroll"),
                financials.get("top_supplier_concentration"),
            ),
        )
        return True
    except Exception as exc:
        logger.warning("smb.store_financials_failed entity=%s error=%s", entity_id, exc)
        return False


# ---------------------------------------------------------------------------
# Cash Flow Forecast
# ---------------------------------------------------------------------------

def _load_monthly_history(entity_id: str, n: int = 12) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT month_end, revenue, cogs, operating_expenses,
                   net_income, cash_balance, payroll
            FROM smb_monthly_financials
            WHERE entity_id = %s
            ORDER BY month_end DESC
            LIMIT %s
            """,
            (entity_id, n),
        )
    except Exception as exc:
        logger.warning("smb.load_history_failed entity=%s error=%s", entity_id, exc)
        return []

    return [
        {
            "month_end": r[0],
            "revenue": float(r[1]) if r[1] is not None else None,
            "cogs": float(r[2]) if r[2] is not None else None,
            "operating_expenses": float(r[3]) if r[3] is not None else None,
            "net_income": float(r[4]) if r[4] is not None else None,
            "cash_balance": float(r[5]) if r[5] is not None else None,
            "payroll": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]


def _ols_trend(values: List[float]) -> float:
    """Return OLS slope (change per period) for a list of values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def forecast_cash_flow(entity_id: str, horizon_months: int = 12) -> Dict[str, Any]:
    """Forecast monthly cash flow for the next horizon_months.

    Method:
    - Load up to 12 months of history (most recent first → reverse for trend)
    - Compute OLS revenue trend and cost trend
    - Project each month: revenue = last + trend × k, costs = last_costs + cost_trend × k
    - Net cash flow = projected_revenue − projected_costs
    - Cumulative cash = latest_cash_balance + sum(net_cash_flows)
    - Cash runway = months until cumulative cash < 0 (None if always positive)
    """
    history = _load_monthly_history(entity_id, n=12)

    if not history:
        return {
            "status": "no_data",
            "entity_id": entity_id,
            "forecast": [],
            "cash_runway_months": None,
        }

    # Reverse: oldest first for trend
    history_asc = list(reversed(history))

    revenues = [p["revenue"] for p in history_asc if p["revenue"] is not None]
    last_revenue = revenues[-1] if revenues else 0.0
    revenue_trend = _ols_trend(revenues) if len(revenues) >= 2 else 0.0

    # Monthly total cost = cogs + opex + payroll
    costs = []
    for p in history_asc:
        c = (p.get("cogs") or 0.0) + (p.get("operating_expenses") or 0.0) + (p.get("payroll") or 0.0)
        if c > 0:
            costs.append(c)
    last_cost = costs[-1] if costs else 0.0
    cost_trend = _ols_trend(costs) if len(costs) >= 2 else 0.0

    latest_cash = history[0].get("cash_balance") or 0.0
    latest_month = history[0]["month_end"]

    cumulative_cash = latest_cash
    cash_runway: Optional[int] = None
    forecast_periods = []

    for k in range(1, horizon_months + 1):
        proj_revenue = max(0.0, last_revenue + revenue_trend * k)
        proj_cost    = max(0.0, last_cost    + cost_trend    * k)
        net_cf = proj_revenue - proj_cost
        cumulative_cash += net_cf

        if cash_runway is None and cumulative_cash < 0:
            cash_runway = k - 1

        month = _add_months(latest_month, k)
        forecast_periods.append({
            "month": month.strftime("%Y-%m"),
            "projected_revenue": round(proj_revenue, 1),
            "projected_costs": round(proj_cost, 1),
            "net_cash_flow": round(net_cf, 1),
            "cumulative_cash": round(cumulative_cash, 1),
        })

    # Health assessment
    if cash_runway is None:
        runway_status = "healthy"
    elif cash_runway <= 3:
        runway_status = "critical"
    elif cash_runway <= 6:
        runway_status = "warning"
    else:
        runway_status = "caution"

    return {
        "status": "ok",
        "entity_id": entity_id,
        "history_months_used": len(history),
        "latest_cash_balance": round(latest_cash, 1),
        "revenue_monthly_trend": round(revenue_trend, 2),
        "cost_monthly_trend": round(cost_trend, 2),
        "cash_runway_months": cash_runway,
        "runway_status": runway_status,
        "forecast": forecast_periods,
    }


# ---------------------------------------------------------------------------
# Supplier Risk
# ---------------------------------------------------------------------------

def compute_supplier_risks(entity_id: str) -> Dict[str, Any]:
    """Analyse supplier concentration and cash cushion risks.

    Risk signals:
    - top_supplier_concentration > 0.50 → high single-source risk
    - accounts_payable growth > 20% QoQ → possible payment stress
    - cash_balance < 2 months of COGS → low liquidity buffer
    """
    if not db.db_read_enabled():
        return {"status": "db_disabled", "entity_id": entity_id, "risks": []}

    # Load supplier concentration separately (not in _load_monthly_history)
    try:
        conc_rows = db.safe_fetchall(
            """
            SELECT month_end, top_supplier_concentration, accounts_payable, cash_balance, cogs
            FROM smb_monthly_financials
            WHERE entity_id = %s
            ORDER BY month_end DESC
            LIMIT 6
            """,
            (entity_id,),
        )
    except Exception as exc:
        logger.warning("smb.supplier_risk_failed entity=%s error=%s", entity_id, exc)
        conc_rows = []

    if not conc_rows:
        return {
            "status": "no_data",
            "entity_id": entity_id,
            "risks": [],
        }

    risks = []
    latest = conc_rows[0]
    _, latest_conc, latest_ap, latest_cash, latest_cogs = latest
    latest_conc = float(latest_conc) if latest_conc is not None else None
    latest_ap   = float(latest_ap)   if latest_ap   is not None else None
    latest_cash = float(latest_cash) if latest_cash is not None else None
    latest_cogs = float(latest_cogs) if latest_cogs is not None else None

    # Risk 1: High supplier concentration
    if latest_conc is not None and latest_conc > 0.50:
        risks.append({
            "risk_type": "supplier_concentration",
            "severity": "high" if latest_conc > 0.70 else "medium",
            "value": round(latest_conc, 2),
            "description": f"Top supplier accounts for {round(latest_conc*100,1)}% of COGS",
        })

    # Risk 2: AP growth (compare latest vs prior if available)
    if len(conc_rows) >= 2 and latest_ap is not None:
        prior_ap = float(conc_rows[1][2]) if conc_rows[1][2] is not None else None
        if prior_ap and prior_ap > 0:
            ap_growth = (latest_ap - prior_ap) / prior_ap
            if ap_growth > 0.20:
                risks.append({
                    "risk_type": "accounts_payable_growth",
                    "severity": "high" if ap_growth > 0.40 else "medium",
                    "value": round(ap_growth, 2),
                    "description": f"Accounts payable grew {round(ap_growth*100,1)}% MoM — possible payment stress",
                })

    # Risk 3: Low cash buffer (< 2 months of COGS)
    if latest_cash is not None and latest_cogs is not None and latest_cogs > 0:
        cash_buffer_months = latest_cash / latest_cogs
        if cash_buffer_months < 2.0:
            risks.append({
                "risk_type": "low_cash_buffer",
                "severity": "high" if cash_buffer_months < 1.0 else "medium",
                "value": round(cash_buffer_months, 2),
                "description": f"Cash covers only {round(cash_buffer_months, 1)} months of COGS",
            })

    # Overall risk level
    if any(r["severity"] == "high" for r in risks):
        overall = "high"
    elif risks:
        overall = "medium"
    else:
        overall = "low"

    return {
        "status": "ok",
        "entity_id": entity_id,
        "overall_risk": overall,
        "top_supplier_concentration": latest_conc,
        "cash_buffer_months": round(latest_cash / latest_cogs, 2)
            if latest_cash is not None and latest_cogs and latest_cogs > 0 else None,
        "risk_count": len(risks),
        "risks": risks,
    }
