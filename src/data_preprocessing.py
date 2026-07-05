import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config

logger = logging.getLogger(__name__)


def load_and_clean_data(input_path=None):
    input_path = input_path or config.RAW_DATA_PATH
    df1 = pd.read_csv(input_path)
    df1 = df1.dropna(how="all")
    df1 = df1[~df1.duplicated()]
    # Remove rows where TotalCharges is blank and fix type
    df1 = df1[df1["TotalCharges"] != " "]
    df1["TotalCharges"] = pd.to_numeric(df1["TotalCharges"])
    df1.drop(["customerID"], axis=1, inplace=True)
    return df1


def encode_categorical(df1, encoders=None):
    """Label-encode categorical columns.

    If `encoders` (a dict of column -> fitted LabelEncoder) is provided,
    reuse them (inference path). Otherwise fit a new encoder per column
    and return the fitted encoders alongside the transformed frame
    (training path).
    """
    fit_new = encoders is None
    if fit_new:
        encoders = {}
    for col in df1.select_dtypes(include="object").columns:
        if col == "Churn":
            continue
        if fit_new:
            encoder = LabelEncoder()
            df1[col] = encoder.fit_transform(df1[col])
            encoders[col] = encoder
        else:
            df1[col] = encoders[col].transform(df1[col])
    if "Churn" in df1.columns:
        df1["Churn"] = df1["Churn"].map({"Yes": 1, "No": 0})
    return df1, encoders


if __name__ == "__main__":
    df = load_and_clean_data()
    df, fitted_encoders = encode_categorical(df)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(fitted_encoders, config.ENCODERS_PATH)
    logger.info("Saved label encoders to %s", config.ENCODERS_PATH)
    df.to_csv(config.CLEANED_DATA_PATH, index=False)
    logger.info("Wrote cleaned data to %s", config.CLEANED_DATA_PATH)
