import pandas as pd
import numpy as np

def classify_regime(data, lookback=10):
    price = data["close"]
    returns = price.pct_change().fillna(0.0)
    m = returns.rolling(lookback).mean().fillna(0.0)
    v = returns.rolling(lookback).std().fillna(0.0)

    regime_score = m / (v + 1e-6)
    regime = pd.Series("neutral", index=data.index)
    regime = regime.where(regime_score <= 0.5, "bull")
    regime = regime.where(regime_score >= -0.5, "bear")

    return pd.DataFrame({
        "regime_score": regime_score,
        "regime": regime
    })
