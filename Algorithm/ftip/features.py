import pandas as pd
import numpy as np


class FeatureEngineer:
    """Builds a suite of quantitative features for the FTIP pipeline."""

    def __init__(self, price_windows=None, volume_windows=None):
        self.price_windows = price_windows or [3, 5, 10]
        self.volume_windows = volume_windows or [3, 5, 10]

    def price_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return compute_price_volume_features(data, self.price_windows, self.volume_windows)

    def fundamental_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return compute_fundamental_features(data)

    def sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return compute_sentiment_features(data)

    def crowd_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return compute_crowd_features(data)

    def regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return compute_regime_features(data)

    def build_feature_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        pieces = [
            self.price_volume_features(data),
            self.fundamental_features(data),
            self.sentiment_features(data),
            self.crowd_features(data),
            self.regime_features(data),
        ]
        return pd.concat(pieces, axis=1)


def compute_price_volume_features(data: pd.DataFrame, price_windows, volume_windows):
    features = pd.DataFrame(index=data.index)
    price = data["close"]
    volume = data.get("volume")

    returns = price.pct_change().fillna(0.0)
    features["return_1d"] = returns
    for w in price_windows:
        features[f"mom_{w}"] = price.pct_change(w).fillna(0.0)
        features[f"vol_{w}"] = returns.rolling(w).std().fillna(0.0)
        features[f"trend_{w}"] = price.rolling(w).mean().pct_change().fillna(0.0)
    if volume is not None:
        for w in volume_windows:
            features[f"vol_chg_{w}"] = volume.pct_change(w).fillna(0.0)
            features[f"vol_ma_ratio_{w}"] = (volume / volume.rolling(w).mean()).fillna(1.0)
    return features


def compute_fundamental_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    if "fundamental" in data.columns:
        fundamental = data["fundamental"].fillna(method="ffill").fillna(0)
        features["fundamental_z"] = (fundamental - fundamental.mean()) / (fundamental.std() + 1e-6)
        features["fundamental_growth"] = fundamental.pct_change().fillna(0.0)
    return features


def compute_sentiment_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    sentiment = data.get("sentiment")
    if sentiment is not None:
        sentiment = sentiment.fillna(0.0)
        features["sentiment_score"] = sentiment
        features["sentiment_trend"] = sentiment.rolling(3).mean().diff().fillna(0.0)
    return features


def compute_crowd_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    crowd = data.get("crowd")
    if crowd is not None:
        crowd = crowd.fillna(0.0)
        features["crowd_intensity"] = crowd
        features["crowd_accel"] = crowd.diff().fillna(0.0)
        features["crowd_volatility"] = crowd.rolling(5).std().fillna(0.0)
    return features


def compute_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    return classify_regime(data)


from .regime import classify_regime
from .features_crowd import compute_crowd_features as compute_crowd_features
from .features_fundamentals import compute_fundamental_features as compute_fundamental_features
from .features_sentiment import compute_sentiment_features as compute_sentiment_features
