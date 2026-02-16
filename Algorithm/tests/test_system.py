from ftip.system import FTIPSystem


def test_system_run_all(sample_data):
    system = FTIPSystem()
    results = system.run_all(sample_data)
    for key in [
        "features",
        "labels",
        "scores",
        "structural_alpha",
        "superfactor",
        "backtest",
    ]:
        assert key in results
    assert len(results["features"]) == len(sample_data)
    assert (
        results["backtest"].total_return
        == results["backtest"].equity_curve.iloc[-1] - 1
    )


def test_providers_health_runtime_exists_and_in_openapi() -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    r = client.get("/providers/health")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    for key in ("openai", "massive", "finnhub", "fred", "secedgar"):
        assert key in data["providers"]

    openapi = client.get("/openapi.json").json()
    assert "/providers/health" in openapi.get("paths", {})
