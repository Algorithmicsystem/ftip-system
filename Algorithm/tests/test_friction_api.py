from fastapi.testclient import TestClient

from api.main import app


def test_friction_simulate_endpoint_returns_schema() -> None:
    payload = {
        "cost_model": {"fee_bps": 1, "slippage_bps": 5, "seed": 42},
        "market_state": {
            "date": "2024-01-02",
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 1000000,
        },
        "execution_plan": {
            "symbol": "AAPL",
            "date": "2024-01-02",
            "side": "BUY",
            "notional": 10000,
            "order_type": "MARKET",
        },
    }

    with TestClient(app) as client:
        response = client.post("/friction/simulate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "total_cost" in body
    assert body["fill_status"] in {"FILLED", "PARTIAL", "NONE"}
