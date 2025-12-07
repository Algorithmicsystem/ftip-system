import pandas as pd

from .features import FeatureEngineer
from .labels import generate_labels


class FatTailPipeline:
    """Build feature matrix, labels, and portfolio weights."""

    def __init__(self, horizon: int = 1, threshold: float = 0.0):
        self.horizon = horizon
        self.threshold = threshold
        self.feature_engineer = FeatureEngineer()

    def build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engineer.build_feature_matrix(data)

    def build_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        return generate_labels(data, horizon=self.horizon, threshold=self.threshold)

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        scores = pd.DataFrame(index=features.index)
        volatility_cols = [c for c in features.columns if c.startswith("vol_")]
        momentum_cols = [c for c in features.columns if c.startswith("mom_")]
        sentiment_cols = [c for c in features.columns if c.startswith("sentiment")]
        crowd_cols = [c for c in features.columns if c.startswith("crowd")]

        if volatility_cols:
            scores["risk_score"] = -features[volatility_cols].mean(axis=1)
        else:
            scores["risk_score"] = 0.0

        alpha_components = []
        for cols in (momentum_cols, sentiment_cols, crowd_cols):
            if cols:
                alpha_components.append(features[cols].mean(axis=1))
        if alpha_components:
            scores["alpha_score"] = sum(alpha_components) / len(alpha_components)
        else:
            scores["alpha_score"] = 0.0

        scores["composite"] = scores["alpha_score"] + scores["risk_score"]
        scores["rank"] = scores["composite"].rank(ascending=False, method="first")
        n = len(scores)
        scores["weight"] = (n - scores["rank"] + 1) / (n * (n + 1) / 2)
        return scores

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        features = self.build_features(data)
        labels = self.build_labels(data)
        scores = self.score(features)
        return pd.concat([features, labels, scores], axis=1)


FTIPPipeline = FatTailPipeline
