
import pytest
import pandas as pd
import numpy as np
from env.feature_engineering import init_feature_space

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "close": [10, 20, 30, 40, 50],
        "volume": [100, 200, 300, 400, 500]
    })

def test_zscore_normalization(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "close", "normalize": True, "method": "zscore" }
        ]
    }
    df, cols = init_feature_space(sample_df.copy(), config)
    assert "close" in cols
    assert abs(df["close"].mean()) < 1e-6
    assert abs(df["close"].std() - 1) < 1e-6

def test_rolling_zscore_normalization(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "close", "normalize": True, "method": "rolling_zscore", "window": 3 }
        ]
    }
    df, cols = init_feature_space(sample_df.copy(), config)
    assert "close" in cols
    assert df["close"].isnull().sum() <= 1  # Allow first row to be NaN

def test_minmax_normalization(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "volume", "normalize": True, "method": "minmax" }
        ]
    }
    df, cols = init_feature_space(sample_df.copy(), config)
    assert "volume" in cols
    assert df["volume"].min() >= 0
    assert df["volume"].max() <= 1

def test_log_return_normalization(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "close", "normalize": True, "method": "log_return" }
        ]
    }
    df, cols = init_feature_space(sample_df.copy(), config)
    expected = np.log(sample_df["close"] / sample_df["close"].shift(1)).fillna(0)
    assert np.allclose(df["close"], expected)

def test_percent_change_normalization(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "close", "normalize": True, "method": "percent_change" }
        ]
    }
    df, cols = init_feature_space(sample_df.copy(), config)
    expected = sample_df["close"].pct_change().fillna(0)
    assert np.allclose(df["close"], expected)

def test_no_normalization(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "close", "normalize": False }
        ]
    }
    df, cols = init_feature_space(sample_df.copy(), config)
    assert np.allclose(df["close"], sample_df["close"])

def test_invalid_method_raises(sample_df):
    config = {
        "FEATURES": [
            { "type": "price", "field": "close", "normalize": True, "method": "invalid" }
        ]
    }
    with pytest.raises(ValueError):
        init_feature_space(sample_df.copy(), config)
