import sys
from pathlib import Path

# Ensure imports work when this test is run from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "Algorithm"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from api.prosperity.routes import _bars_required


def test_bars_required_honors_requested_lookback_above_floor() -> None:
    assert _bars_required(250) == 250


def test_bars_required_applies_minimum_quality_floor() -> None:
    assert _bars_required(5) == 30
