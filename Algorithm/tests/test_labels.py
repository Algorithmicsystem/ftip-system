from ftip.labels import create_forward_returns, generate_labels


def test_forward_returns(sample_data):
    fr = create_forward_returns(sample_data["close"], horizon=2)
    assert fr.isna().sum() >= 1
    assert (fr.iloc[:-2] != 0).any()


def test_generate_labels(sample_data):
    labels = generate_labels(sample_data, horizon=1, threshold=0)
    assert set(labels.columns) == {"forward_return", "label"}
    assert labels["label"].isin([0, 1]).all()
