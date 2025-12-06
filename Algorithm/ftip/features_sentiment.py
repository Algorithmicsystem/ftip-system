import pandas as pd

def compute_sentiment_features(data):
    features = pd.DataFrame(index=data.index)
    sentiment = data.get("sentiment")
    if sentiment is not None:
        sentiment = sentiment.fillna(0.0)
        features["sentiment_score"] = sentiment
        features["sentiment_trend"] = sentiment.rolling(3).mean().diff().fillna(0.0)
    return features
