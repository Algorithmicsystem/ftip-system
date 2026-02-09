import pandas as pd
from .features import FeatureEngineer
from .labels import generate_labels


class FatTailPipeline:
    def __init__(self, horizon=1, threshold=0):
        self.horizon = horizon
        self.threshold = threshold
        self.fe = FeatureEngineer()

    def build_features(self, data):
        return self.fe.build_feature_matrix(data)

    def build_labels(self, data):
        return generate_labels(data, horizon=self.horizon, threshold=self.threshold)

    def score(self, features):
        scores = pd.DataFrame(index=features.index)

        vol_cols = [c for c in features if c.startswith("vol_")]
        mom_cols = [c for c in features if c.startswith("mom_")]
        sent_cols = [c for c in features if c.startswith("sentiment")]
        crowd_cols = [c for c in features if c.startswith("crowd")]

        scores["risk_score"] = -features[vol_cols].mean(axis=1) if vol_cols else 0
        alpha_components = []
        for cols in (mom_cols, sent_cols, crowd_cols):
            if cols:
                alpha_components.append(features[cols].mean(axis=1))
        scores["alpha_score"] = (
            sum(alpha_components) / len(alpha_components) if alpha_components else 0
        )

        scores["composite"] = scores["alpha_score"] + scores["risk_score"]
        scores["rank"] = scores["composite"].rank(ascending=False)
        num_scores = len(scores)
        scores["weight"] = (num_scores - scores["rank"] + 1) / (
            num_scores * (num_scores + 1) / 2
        )

        return scores

    def run(self, data):
        features = self.build_features(data)
        labels = self.build_labels(data)
        scores = self.score(features)
        return pd.concat([features, labels, scores], axis=1)


FTIPPipeline = FatTailPipeline
