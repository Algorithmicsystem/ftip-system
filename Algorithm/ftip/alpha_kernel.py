import pandas as pd
import numpy as np

class StructuralAlphaKernel:
    """Extracts structural alpha signals via lightweight PCA."""

    def __init__(self, n_factors: int = 2):
        self.n_factors = n_factors
        self.components_ = None
        self.mean_ = None

    def fit(self, features: pd.DataFrame):
        matrix = features.fillna(0.0).values
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        self.components_ = eigvecs[:, : self.n_factors]
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.components_ is None:
            raise RuntimeError("Kernel must be fit before transform.")
        matrix = features.fillna(0.0).values
        centered = matrix - self.mean_
        scores = centered @ self.components_
        return pd.DataFrame(
            scores,
            index=features.index,
            columns=[f"factor_{i+1}" for i in range(scores.shape[1])]
        )

    def detect_anomalies(self, features: pd.DataFrame) -> pd.Series:
        transformed = self.transform(features)
        zscore = (transformed - transformed.mean()) / (transformed.std() + 1e-6)
        return zscore.abs().mean(axis=1)

    def structural_alpha(self, features: pd.DataFrame) -> pd.Series:
        factors = self.transform(features)
        anomaly_penalty = self.detect_anomalies(features)
        alpha = factors.mean(axis=1) - anomaly_penalty
        return alpha.rename("structural_alpha")
