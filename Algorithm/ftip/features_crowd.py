import pandas as pd


def compute_crowd_features(data):
    features = pd.DataFrame(index=data.index)
    crowd = data.get("crowd")
    if crowd is not None:
        crowd = crowd.fillna(0.0)
        features["crowd_intensity"] = crowd
        features["crowd_accel"] = crowd.diff().fillna(0.0)
        features["crowd_volatility"] = crowd.rolling(5).std().fillna(0.0)
    return features
