import json
import logging

import joblib
import pandas as pd

import config
from data_preprocessing import encode_categorical
from feature_engineering import add_features, scale_numerical

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_encoders = None
_feature_columns = None


def load_artifacts(force_reload=False):
    """Load model + preprocessing artifacts once and cache them."""
    global _model, _scaler, _encoders, _feature_columns
    if force_reload or _model is None:
        _model = joblib.load(config.MODEL_PATH)
        _scaler = joblib.load(config.SCALER_PATH)
        _encoders = joblib.load(config.ENCODERS_PATH)
        with open(config.FEATURE_COLUMNS_PATH) as f:
            _feature_columns = json.load(f)
        logger.info("Loaded model, scaler, encoders, and feature columns")
    return _model, _scaler, _encoders, _feature_columns


def predict_churn(input_dict):
    model, scaler, encoders, feature_columns = load_artifacts()

    df = pd.DataFrame([input_dict])
    df, _ = encode_categorical(df, encoders=encoders)
    df = add_features(df)
    df, _ = scale_numerical(df, scaler=scaler)
    df = df[feature_columns]

    prediction = model.predict(df)
    probability = model.predict_proba(df)[0, 1]
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability),
    }


# Example usage
if __name__ == "__main__":
    sample_input = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 1680.0,
    }
    logging.getLogger(__name__).info(predict_churn(sample_input))
