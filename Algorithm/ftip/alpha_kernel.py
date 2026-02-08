import pandas as pd
import numpy as np


class StructuralAlphaKernel:
    """Extracts structural alpha signals via lightweight PCA."""

    def __init__(self, n_factors: int = 2):
        self.n_factors = n_factors
        self.components_ = None
        self.mean_ = None
        # Keep track of which columns are used in the numeric matrix
        self.feature_columns_ = None

    def fit(self, features: pd.DataFrame):
        """
        Fit a simple PCA-style kernel on the numeric features.

        Non-numeric columns (like regime labels) are dropped so that
        numpy operations don't choke on strings.
        """
        # Only keep numeric columns for the kernel
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0.0)
        self.feature_columns_ = numeric_features.columns

        matrix = numeric_features.values
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_

        # Simple covariance and eigen-decomposition
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        # Keep the top n_factors eigenvectors
        self.components_ = eigvecs[:, : self.n_factors]
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Project features into factor space using the learned components.
        """
        if self.components_ is None:
            raise RuntimeError("Kernel must be fit before transform.")

        # Rebuild the numeric matrix with the same columns used in fit()
        matrix = features.reindex(columns=self.feature_columns_).fillna(0.0).values

        centered = matrix - self.mean_
        scores = centered @ self.components_

        return pd.DataFrame(
            scores,
            index=features.index,
            columns=[f"factor_{i + 1}" for i in range(scores.shape[1])],
        )

    def detect_anomalies(self, features: pd.DataFrame) -> pd.Series:
        """
        Compute an anomaly score based on factor z-scores.
        """
        transformed = self.transform(features)
        zscore = (transformed - transformed.mean()) / (transformed.std() + 1e-6)
        return zscore.abs().mean(axis=1)

    def structural_alpha(self, features: pd.DataFrame) -> pd.Series:
        """
        Combine factor scores and anomaly penalty into a single alpha signal.
        """
        factors = self.transform(features)
        anomaly_penalty = self.detect_anomalies(features)
        alpha = factors.mean(axis=1) - anomaly_penalty
        return alpha.rename("structural_alpha")
