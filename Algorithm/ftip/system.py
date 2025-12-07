import pandas as pd

from .pipeline import FatTailPipeline
from .alpha_kernel import StructuralAlphaKernel
from .superfactor import SuperfactorModel
from .backtest.simulator import evaluate_backtest


class FTIPSystem:
    """High level orchestrator for the FTIP workflow."""

    def __init__(self, pipeline: FatTailPipeline | None = None):
        self.pipeline = pipeline or FatTailPipeline()
        self.kernel = StructuralAlphaKernel()
        self.superfactor = SuperfactorModel(self.kernel)

    def run_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.build_features(data)

    def run_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        features = self.run_features(data)
        labels = self.pipeline.build_labels(data)
        scores = self.pipeline.score(features)
        return pd.concat([features, labels, scores], axis=1)

    def run_kernel(self, data: pd.DataFrame) -> pd.Series:
        features = self.run_features(data)
        self.kernel.fit(features)
        return self.kernel.structural_alpha(features)

    def run_superfactor(self, data: pd.DataFrame) -> pd.Series:
        features = self.run_features(data)
        return self.superfactor.fit_transform(features)

    def run_backtest(self, data: pd.DataFrame) -> pd.Series:
        scored = self.run_scores(data)
        weights = scored["weight"]
        return evaluate_backtest(data["close"], weights)

    def run_all(self, data: pd.DataFrame):
        features = self.run_features(data)
        labels = self.pipeline.build_labels(data)
        scores = self.pipeline.score(features)
        structural_alpha = self.kernel.fit(features).structural_alpha(features)
        superfactor = self.superfactor.fit_transform(features)
        backtest = evaluate_backtest(data["close"], scores["weight"])
        return {
            "features": features,
            "labels": labels,
            "scores": scores,
            "structural_alpha": structural_alpha,
            "superfactor": superfactor,
            "backtest": backtest,
        }
