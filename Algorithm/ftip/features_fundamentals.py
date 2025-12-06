import pandas as pd

def compute_fundamental_features(data):
    features = pd.DataFrame(index=data.index)
    if "fundamental" in data.columns:
        f = data["fundamental"].fillna(method="ffill").fillna(0)
        features["fundamental_z"] = (f - f.mean()) / (f.std() + 1e-6)
        features["fundamental_growth"] = f.pct_change().fillna(0.0)
    return features
