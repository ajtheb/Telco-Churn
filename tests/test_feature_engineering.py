import numpy as np
import pandas as pd
import pytest

from feature_engineering import NUMERICAL_COLUMNS, add_features, scale_numerical


@pytest.fixture
def numeric_df():
    return pd.DataFrame(
        {
            "tenure": [0, 12, 24, 36, 48, 60, 72],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0, 100.0, 50.0, 30.0],
            "TotalCharges": [19.0, 480.0, 1440.0, 2880.0, 4800.0, 3000.0, 2160.0],
        }
    )


def test_add_features_tenure_buckets_and_ratio(numeric_df):
    df = add_features(numeric_df.copy())

    # bins=[0,12,24,36,48,60,72], labels=False, right-closed: 0 falls outside
    # every bin (NaN); 12/24/36/48/60/72 land in buckets 0..5.
    expected_groups = [np.nan, 0, 1, 2, 3, 4, 5]
    assert df["tenure_group"].tolist()[1:] == expected_groups[1:]
    assert pd.isna(df["tenure_group"].iloc[0])

    expected_ratio = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    assert (df["monthly_total_ratio"] == expected_ratio).all()


def test_scale_numerical_fits_when_no_scaler_given(numeric_df):
    df = add_features(numeric_df.copy())
    scaled_df, scaler = scale_numerical(df.copy())

    means = scaled_df[NUMERICAL_COLUMNS].mean()
    stds = scaled_df[NUMERICAL_COLUMNS].std(ddof=0)
    assert np.allclose(means, 0, atol=1e-8)
    assert np.allclose(stds, 1, atol=1e-8)


def test_scale_numerical_transform_only_does_not_refit(numeric_df):
    df = add_features(numeric_df.copy())
    _, fitted_scaler = scale_numerical(df.copy())
    original_mean = fitted_scaler.mean_.copy()

    other_df = add_features(numeric_df.copy() * 2)
    raw_values = other_df[NUMERICAL_COLUMNS].copy()
    scaled_other, returned_scaler = scale_numerical(other_df, scaler=fitted_scaler)

    assert returned_scaler is fitted_scaler
    assert np.array_equal(fitted_scaler.mean_, original_mean)
    manual = (raw_values - original_mean) / fitted_scaler.scale_
    assert np.allclose(scaled_other[NUMERICAL_COLUMNS].to_numpy(), manual.to_numpy())
