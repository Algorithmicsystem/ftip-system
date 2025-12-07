import pandas as pd
import numpy as np


def classify_regime(data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    price = data["close"]
    returns = price.pct_change().fillna(0.0)
    rolling_mean = returns.rolling(lookback).mean().fillna(0.0)
    rolling_vol = returns.rolling(lookback).std().fillna(0.0)

    regime_score = rolling_mean / (rolling_vol + 1e-6)
    regime = pd.Series("neutral", index=data.index)
    regime = regime.where(regime_score <= 0.5, "bull")
    regime = regime.where(regime_score >= -0.5, "bear")

    features = pd.DataFrame(index=data.index)
    features["regime_score"] = regime_score
    features["regime"] = regime
    return features
