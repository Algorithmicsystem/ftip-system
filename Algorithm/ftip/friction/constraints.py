from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, Tuple

from .models import ConstraintConfig, ExecutionPlan


class ConstraintsEngine:
    def __init__(self, config: ConstraintConfig):
        self.config = config

    def validate_trade(
        self,
        plan: ExecutionPlan,
        *,
        current_position_weight: float = 0.0,
        gross_exposure: float = 0.0,
        turnover: float = 0.0,
        symbol_metadata: Optional[Dict[str, Any]] = None,
        sector_weights: Optional[Dict[str, float]] = None,
        beta_exposure: Optional[float] = None,
        last_trade_date: Optional[dt.date] = None,
    ) -> Tuple[bool, str]:
        metadata = symbol_metadata or {}
        if plan.notional < self.config.min_trade_notional:
            return False, "TRADE_BELOW_MIN_NOTIONAL"
        if (
            self.config.max_trade_notional is not None
            and plan.notional > self.config.max_trade_notional
        ):
            return False, "TRADE_ABOVE_MAX_NOTIONAL"
        if abs(current_position_weight) > self.config.max_position_weight:
            return False, "POSITION_WEIGHT_LIMIT"
        if gross_exposure > self.config.max_gross_exposure:
            return False, "GROSS_EXPOSURE_LIMIT"
        if turnover > self.config.max_turnover_per_rebalance:
            return False, "TURNOVER_LIMIT"
        if last_trade_date and self.config.cooldown_days > 0:
            if (plan.date - last_trade_date).days < self.config.cooldown_days:
                return False, "COOLDOWN_ACTIVE"

        if self.config.max_sector_weight is not None and sector_weights is not None:
            sector = metadata.get("sector")
            if (
                sector is not None
                and sector_weights.get(sector, 0.0) > self.config.max_sector_weight
            ):
                return False, "SECTOR_WEIGHT_LIMIT"

        if self.config.max_beta_exposure is not None and beta_exposure is not None:
            if abs(beta_exposure) > self.config.max_beta_exposure:
                return False, "BETA_EXPOSURE_LIMIT"

        return True, "OK"
