import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)

NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "monthly_total_ratio",
]


def add_features(df1):
    # Tenure buckets
    df1["tenure_group"] = pd.cut(
        df1["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=False
    )
    # Monthly/Total Ratio
    df1["monthly_total_ratio"] = (
        df1["MonthlyCharges"] / (df1["TotalCharges"] + 1)
    )
    return df1


def scale_numerical(df1, scaler=None):
    """Scale numerical columns.

    If `scaler` is provided, only transform with it (inference path).
    Otherwise fit a new StandardScaler and return it alongside the
    transformed frame (training path).
    """
    fit_new = scaler is None
    if fit_new:
        scaler = StandardScaler()
        df1[NUMERICAL_COLUMNS] = scaler.fit_transform(df1[NUMERICAL_COLUMNS])
    else:
        df1[NUMERICAL_COLUMNS] = scaler.transform(df1[NUMERICAL_COLUMNS])
    return df1, scaler


if __name__ == "__main__":
    df = pd.read_csv(config.CLEANED_DATA_PATH)
    df = add_features(df)
    df, fitted_scaler = scale_numerical(df)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.FEATURED_DATA_PATH), exist_ok=True)
    joblib.dump(fitted_scaler, config.SCALER_PATH)
    logger.info("Saved scaler to %s", config.SCALER_PATH)
    df.to_csv(config.FEATURED_DATA_PATH, index=False)
    logger.info("Wrote featured data to %s", config.FEATURED_DATA_PATH)
