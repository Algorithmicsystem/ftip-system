from __future__ import annotations

from .costs import bps_to_cash
from .fills import compute_fill_fraction
from .impact import compute_impact_paid
from .models import CostModel, ExecutionPlan, ExecutionResult, MarketStateInputs
from .slippage import compute_slippage_paid, compute_spread_paid


class FrictionEngine:
    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def simulate(
        self,
        market: MarketStateInputs,
        plan: ExecutionPlan,
        *,
        apply_overnight: bool = False,
        held_notional: float = 0.0,
    ) -> ExecutionResult:
        fill_fraction, reason = compute_fill_fraction(self.cost_model, market, plan)
        filled_notional = plan.notional * fill_fraction
        if filled_notional <= 0:
            return ExecutionResult(
                filled_qty=0.0,
                filled_notional=0.0,
                avg_fill_price=market.close,
                fees_paid=0.0,
                slippage_paid=0.0,
                impact_paid=0.0,
                spread_paid=0.0,
                overnight_risk_paid=0.0,
                total_cost=0.0,
                effective_price=market.close,
                fill_status="NONE",
                reason=reason,
            )

        filled_qty = filled_notional / market.close
        fees_paid = bps_to_cash(filled_notional, self.cost_model.fee_bps)
        slippage_paid = compute_slippage_paid(filled_notional, self.cost_model, market)
        impact_paid = compute_impact_paid(filled_notional, self.cost_model, market)
        spread_paid = compute_spread_paid(filled_notional, market)
        overnight_base = held_notional if held_notional > 0 else filled_notional
        overnight_risk_paid = (
            bps_to_cash(overnight_base, self.cost_model.overnight_gap_risk_bps)
            if apply_overnight
            else 0.0
        )
        total_cost = (
            fees_paid + slippage_paid + impact_paid + spread_paid + overnight_risk_paid
        )
        avg_fill_price = market.close
        direction = 1.0 if plan.side == "BUY" else -1.0
        effective_price = avg_fill_price + direction * (
            total_cost / max(filled_qty, 1e-12)
        )
        fill_status = "FILLED" if fill_fraction >= 1.0 else "PARTIAL"

        return ExecutionResult(
            filled_qty=filled_qty,
            filled_notional=filled_notional,
            avg_fill_price=avg_fill_price,
            fees_paid=fees_paid,
            slippage_paid=slippage_paid,
            impact_paid=impact_paid,
            spread_paid=spread_paid,
            overnight_risk_paid=overnight_risk_paid,
            total_cost=total_cost,
            effective_price=effective_price,
            fill_status=fill_status,
            reason=reason,
        )
