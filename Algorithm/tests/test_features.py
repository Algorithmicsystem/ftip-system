from ftip.features import FeatureEngineer


def test_feature_engineer_builds_expected_columns(sample_data):
    fe = FeatureEngineer(price_windows=[2], volume_windows=[2])
    features = fe.build_feature_matrix(sample_data)
    expected = {
        "return_1d",
        "mom_2",
        "vol_2",
        "trend_2",
        "vol_chg_2",
        "vol_ma_ratio_2",
        "fundamental_z",
        "fundamental_growth",
        "sentiment_score",
        "sentiment_trend",
        "crowd_intensity",
        "crowd_accel",
        "crowd_volatility",
        "regime_score",
        "regime",
    }
    assert expected.issubset(set(features.columns))
    assert len(features) == len(sample_data)
