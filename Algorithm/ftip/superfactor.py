import pandas as pd

from .alpha_kernel import StructuralAlphaKernel

def compute_momentum(data: pd.DataFrame, window: int = 5) -> pd.Series:
    return data["close"].pct_change(window).fillna(0.0)

def compute_superfactor(
    alpha_series: pd.Series,
    sentiment: pd.Series,
    crowd_accel: pd.Series,
    regime_score: pd.Series,
    momentum: pd.Series,
) -> pd.Series:
    aligned = pd.concat(
        [alpha_series, sentiment, crowd_accel, regime_score, momentum],
        axis=1
    ).fillna(0.0)
    weights = [0.35, 0.2, 0.15, 0.15, 0.15]
    superfactor = aligned.mul(weights).sum(axis=1)
    return superfactor.rename("superfactor_alpha")

class SuperfactorModel:
    """Combine multiple alpha sources into a single superfactor."""

    def __init__(self, kernel=None):
        self.kernel = kernel or StructuralAlphaKernel()

    def fit_transform(self, features: pd.DataFrame) -> pd.Series:
        self.kernel.fit(features)
        structural_alpha = self.kernel.structural_alpha(features)
        sentiment = features.get("sentiment_score", pd.Series(0.0, index=features.index))
        crowd_accel = features.get("crowd_accel", pd.Series(0.0, index=features.index))
        regime_score = features.get("regime_score", pd.Series(0.0, index=features.index))
        momentum = features.get("mom_5", pd.Series(0.0, index=features.index))
        return compute_superfactor(structural_alpha, sentiment, crowd_accel, regime_score, momentum)
