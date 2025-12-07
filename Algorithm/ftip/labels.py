import pandas as pd


def create_forward_returns(price: pd.Series, horizon: int = 1) -> pd.Series:
    return price.shift(-horizon) / price - 1.0


def generate_labels(data: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
    forward_returns = create_forward_returns(data["close"], horizon)
    labels = (forward_returns > threshold).astype(int)
    return pd.DataFrame({"forward_return": forward_returns, "label": labels})
