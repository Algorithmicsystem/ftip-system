from fastapi.testclient import TestClient

from api.main import app


def test_backtest_endpoints(monkeypatch):
    from api.backtest import service

    monkeypatch.setattr(
        service,
        "run_backtest",
        lambda **_kwargs: {"run_id": "run-123", "status": "success"},
    )
    monkeypatch.setattr(
        service,
        "fetch_results",
        lambda _run_id: {
            "run": {"id": "run-123"},
            "metrics": {
                "cagr": 0.1,
                "sharpe": 1.0,
                "sortino": 1.2,
                "maxdd": -0.2,
                "turnover": 0.5,
            },
            "regime_metrics": [],
            "summary": {"beats_spy": True},
        },
    )
    monkeypatch.setattr(
        service,
        "fetch_equity_curve",
        lambda _run_id: [
            {
                "dt": "2024-01-02",
                "equity": 1.0,
                "drawdown": 0.0,
                "benchmark_equity": 1.0,
            }
        ],
    )

    with TestClient(app) as client:
        run_resp = client.post(
            "/backtest/run",
            json={
                "symbol": "AAPL",
                "universe": "custom",
                "date_start": "2024-01-01",
                "date_end": "2024-01-10",
                "horizon": "swing",
                "risk_mode": "balanced",
                "signal_version_hash": "auto",
                "cost_model": {"fee_bps": 1, "slippage_bps": 5},
            },
        )
        assert run_resp.status_code == 200
        run_id = run_resp.json()["run_id"]

        res_resp = client.get(f"/backtest/results?run_id={run_id}")
        assert res_resp.status_code == 200
        assert res_resp.json()["run"]["id"] == "run-123"

        curve_resp = client.get(f"/backtest/equity-curve?run_id={run_id}")
        assert curve_resp.status_code == 200
        assert curve_resp.json()["points"]
