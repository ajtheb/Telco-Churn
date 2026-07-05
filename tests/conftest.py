import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import model_inference  # noqa: E402
from data_preprocessing import encode_categorical, load_and_clean_data  # noqa: E402
from feature_engineering import add_features, scale_numerical  # noqa: E402


@pytest.fixture
def sample_raw_df():
    """Small synthetic frame shaped like the raw Telco CSV.

    Row 3 and row 4 are exact duplicates (dedup check), row 1 has a
    blank TotalCharges (blank-row-drop check).
    """
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003", "0004", "0004"],
            "gender": ["Female", "Male", "Male", "Female", "Female"],
            "SeniorCitizen": [0, 1, 0, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes", "No", "No"],
            "tenure": [1, 34, 2, 45, 45],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": [
                "No phone service", "No", "No", "No", "No",
            ],
            "InternetService": ["DSL", "DSL", "DSL", "DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "Yes"],
            "OnlineBackup": ["Yes", "No", "Yes", "No", "No"],
            "DeviceProtection": ["No", "Yes", "No", "Yes", "Yes"],
            "TechSupport": ["No", "No", "No", "No", "No"],
            "StreamingTV": ["No", "No", "No", "No", "No"],
            "StreamingMovies": ["No", "No", "No", "No", "No"],
            "Contract": [
                "Month-to-month", "One year", "Month-to-month", "One year", "One year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check", "Mailed check", "Electronic check",
                "Mailed check", "Mailed check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 42.30],
            "TotalCharges": ["29.85", " ", "108.15", "1840.75", "1840.75"],
            "Churn": ["No", "No", "Yes", "No", "No"],
        }
    )


@pytest.fixture
def trained_artifacts(sample_raw_df, tmp_path, monkeypatch):
    """Train a tiny model on the fixture data and point config at its artifacts."""
    raw_csv = tmp_path / "raw.csv"
    sample_raw_df.to_csv(raw_csv, index=False)

    df = load_and_clean_data(input_path=str(raw_csv))
    df, encoders = encode_categorical(df)
    df = add_features(df)
    df, scaler = scale_numerical(df)

    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(x, y)

    joblib.dump(clf, tmp_path / "model.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    joblib.dump(encoders, tmp_path / "encoders.pkl")
    (tmp_path / "feature_columns.json").write_text(json.dumps(list(x.columns)))

    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(config, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
    monkeypatch.setattr(config, "ENCODERS_PATH", str(tmp_path / "encoders.pkl"))
    monkeypatch.setattr(
        config, "FEATURE_COLUMNS_PATH", str(tmp_path / "feature_columns.json")
    )

    model_inference.load_artifacts(force_reload=True)
