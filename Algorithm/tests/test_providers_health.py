import os

import pytest

from api.providers.finnhub import FinnhubProvider
from api.providers.fred import FREDProvider
from api.providers.sec_edgar import SecEdgarProvider


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_finnhub_health_integration_runs_only_with_key():
    if not os.getenv("FINNHUB_API_KEY"):
        pytest.skip("FINNHUB_API_KEY not set")

    health = FinnhubProvider().health_check()
    assert health.enabled is True
    assert health.status in {"ok", "down"}


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_fred_health_integration_runs_only_with_key():
    if not os.getenv("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set")

    health = FREDProvider().health_check()
    assert health.enabled is True
    assert health.status in {"ok", "down"}


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_sec_edgar_health_integration():
    health = SecEdgarProvider().health_check()
    assert health.enabled is True
    assert health.status in {"ok", "degraded", "down"}
