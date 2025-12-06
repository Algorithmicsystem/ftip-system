import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure Algorithm/ is on sys.path so "import ftip" works when pytest runs from repo root
ROOT = Path(__file__).resolve().parents[1]  # Algorithm
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def sample_data():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    close = pd.Series(np.linspace(100, 120, len(idx)) + np.sin(np.arange(len(idx))), index=idx)
    volume = pd.Series(1_000 + np.arange(len(idx)) * 10, index=idx)
    fundamental = pd.Series(np.linspace(10, 15, len(idx)), index=idx)
    sentiment = pd.Series(np.sin(np.linspace(0, 3.14, len(idx))), index=idx)
    crowd = pd.Series(np.cos(np.linspace(0, 3.14, len(idx))) * 0.5 + 0.5, index=idx)

    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "fundamental": fundamental,
        "sentiment": sentiment,
        "crowd": crowd,
    })
