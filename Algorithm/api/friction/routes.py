from fastapi import APIRouter
from pydantic import BaseModel

from api.friction.models import (
    CostModel,
    ExecutionPlan,
    ExecutionResult,
    MarketStateInputs,
)
from ftip.friction.engine import FrictionEngine

router = APIRouter(prefix="/friction", tags=["friction"])


class FrictionSimulateRequest(BaseModel):
    cost_model: CostModel
    market_state: MarketStateInputs
    execution_plan: ExecutionPlan


@router.post("/simulate", response_model=ExecutionResult)
def simulate_friction(req: FrictionSimulateRequest) -> ExecutionResult:
    engine = FrictionEngine(req.cost_model)
    return engine.simulate(req.market_state, req.execution_plan)
