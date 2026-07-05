import logging
import os

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

RAW_DATA_PATH = os.environ.get(
    "RAW_DATA_PATH", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
CLEANED_DATA_PATH = os.environ.get(
    "CLEANED_DATA_PATH", "data/processed/cleaned_telco.csv"
)
FEATURED_DATA_PATH = os.environ.get(
    "FEATURED_DATA_PATH", "data/processed/featured_telco.csv"
)
X_TEST_PATH = os.environ.get("X_TEST_PATH", "data/processed/X_test.csv")
Y_TEST_PATH = os.environ.get("Y_TEST_PATH", "data/processed/y_test.csv")

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_rf.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

RUN_ID_PATH = os.environ.get("RUN_ID_PATH", "run_id.txt")

MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "telco-churn")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
