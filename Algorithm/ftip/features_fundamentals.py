import pandas as pd


def compute_fundamental_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    if "fundamental" in data.columns:
        fundamental = data["fundamental"].fillna(method="ffill").fillna(0)
        features["fundamental_z"] = (fundamental - fundamental.mean()) / (fundamental.std() + 1e-6)
        features["fundamental_growth"] = fundamental.pct_change().fillna(0.0)
    return features
