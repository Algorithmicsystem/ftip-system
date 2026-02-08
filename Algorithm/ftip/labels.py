import pandas as pd


def create_forward_returns(price, horizon=1):
    return price.shift(-horizon) / price - 1


def generate_labels(data, horizon=1, threshold=0):
    fr = create_forward_returns(data["close"], horizon)
    labels = (fr > threshold).astype(int)
    return pd.DataFrame({"forward_return": fr, "label": labels})
