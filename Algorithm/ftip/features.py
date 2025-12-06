import pandas as pd
import numpy as np

from .features_crowd import compute_crowd_features
from .features_fundamentals import compute_fundamental_features
from .features_sentiment import compute_sentiment_features
from .regime import classify_regime


class FeatureEngineer:
    def __init__(self, price_windows=None, volume_windows=None):
        self.price_windows = price_windows or [3, 5, 10]
        self.volume_windows = volume_windows or [3, 5, 10]

    def price_volume_features(self, data):
        return compute_price_volume_features(data, self.price_windows, self.volume_windows)

    def fundamental_features(self, data):
        return compute_fundamental_features(data)

    def sentiment_features(self, data):
        return compute_sentiment_features(data)

    def crowd_features(self, data):
        return compute_crowd_features(data)

    def regime_features(self, data):
        return classify_regime(data)

    def build_feature_matrix(self, data):
        return pd.concat([
            self.price_volume_features(data),
            self.fundamental_features(data),
            self.sentiment_features(data),
            self.crowd_features(data),
            self.regime_features(data),
        ], axis=1)


def compute_price_volume_features(data, price_windows, volume_windows):
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
