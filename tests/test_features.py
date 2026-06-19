from src import config
from src.features import build_features, feature_columns


def test_no_nans(feat):
    assert not feat[feature_columns(feat)].isna().any().any()


def test_weather_features_present(feat):
    for col in ("temp_c", "cooling_deg", "heating_deg"):
        assert col in feat.columns


def test_lags_are_leak_safe(raw):
    """lag_24 at row i must equal the target HORIZON hours earlier."""
    feat = build_features(raw)
    h = config.HORIZON_HOURS
    merged = raw.set_index("timestamp")[config.TARGET]
    sample = feat.iloc[100]
    earlier = sample["timestamp"] - __import__("pandas").Timedelta(hours=h)
    assert abs(sample[f"lag_{h}"] - merged.loc[earlier]) < 1e-6
