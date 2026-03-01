import datetime as dt

from ftip.friction import CostModel, ExecutionPlan, FrictionEngine, MarketStateInputs
from ftip.friction.impact import compute_impact_bps
from ftip.friction.slippage import compute_slippage_bps


def _market(rv_20: float = 0.02) -> MarketStateInputs:
    return MarketStateInputs(
        date=dt.date(2024, 1, 2),
        open=100,
        high=102,
        low=99,
        close=101,
        volume=1_000_000,
        adv_20=900_000,
        rv_20=rv_20,
    )


def test_friction_deterministic_given_seed() -> None:
    model = CostModel(seed=7)
    plan = ExecutionPlan(
        symbol="AAPL",
        date=dt.date(2024, 1, 2),
        side="BUY",
        notional=50_000,
        order_type="LIMIT",
        limit_price=101,
    )
    engine = FrictionEngine(model)
    first = engine.simulate(_market(), plan)
    second = engine.simulate(_market(), plan)
    assert first.model_dump() == second.model_dump()


def test_slippage_and_impact_increase_with_volatility() -> None:
    model = CostModel()
    low_vol = _market(rv_20=0.01)
    high_vol = _market(rv_20=0.05)
    assert compute_slippage_bps(model, high_vol) >= compute_slippage_bps(model, low_vol)
    assert compute_impact_bps(100_000, model, high_vol) >= compute_impact_bps(
        100_000, model, low_vol
    )


def test_adv_constraint_causes_partial_or_none_fill() -> None:
    model = CostModel(max_adv_pct=0.01)
    plan = ExecutionPlan(
        symbol="AAPL",
        date=dt.date(2024, 1, 2),
        side="BUY",
        notional=5_000_000,
        order_type="MARKET",
    )
    result = FrictionEngine(model).simulate(_market(), plan)
    assert result.fill_status in {"PARTIAL", "NONE"}
    assert "ADV" in result.reason


def test_limit_fill_logic_for_buy_orders() -> None:
    model = CostModel(seed=11)
    market = _market()
    engine = FrictionEngine(model)

    below_low = ExecutionPlan(
        symbol="AAPL",
        date=dt.date(2024, 1, 2),
        side="BUY",
        notional=10_000,
        order_type="LIMIT",
        limit_price=98,
    )
    above_high = below_low.model_copy(update={"limit_price": 103})
    inside = below_low.model_copy(update={"limit_price": 100})

    assert engine.simulate(market, below_low).fill_status == "NONE"
    assert engine.simulate(market, above_high).fill_status == "FILLED"
    # probabilistic path is deterministic due to stable seed+hash
    first_inside = engine.simulate(market, inside)
    second_inside = engine.simulate(market, inside)
    assert first_inside.model_dump() == second_inside.model_dump()
