"""FTIP system public interface."""

from .features import FeatureEngineer
from .pipeline import FatTailPipeline, FatTailPipeline as FTIPPipeline
from .backtest.simulator import (
    Portfolio,
    BacktestSimulator,
    BacktestResult,
    evaluate_backtest,
)
from .system import FTIPSystem
from .labels import create_forward_returns, generate_labels

__all__ = [
    "FeatureEngineer",
    "FatTailPipeline",
    "FTIPPipeline",
    "Portfolio",
    "BacktestSimulator",
    "BacktestResult",
    "evaluate_backtest",
    "FTIPSystem",
    "create_forward_returns",
    "generate_labels",
]
