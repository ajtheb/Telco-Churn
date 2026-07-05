import json
import logging
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import config

logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.FEATURED_DATA_PATH), exist_ok=True)


def train_model(
    data_path=None,
    model_out_path=None,
    mlflow_experiment=None,
    n_estimators=200,
    test_size=0.2,
    random_state=42,
):
    data_path = data_path or config.FEATURED_DATA_PATH
    model_out_path = model_out_path or config.MODEL_PATH
    mlflow_experiment = mlflow_experiment or config.MLFLOW_EXPERIMENT_NAME

    logger.debug("Current working directory: %s", os.getcwd())
    logger.debug("Tracking URI: %s", mlflow.get_tracking_uri())

    # 1. Load Data
    df = pd.read_csv(data_path)
    x = df.drop(["Churn"], axis=1)
    y = df["Churn"]

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(list(x.columns), f)

    # 2. Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # 3. Define Model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
        )

    # 4. Train Model with MLflow Tracking
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run() as run:
        clf.fit(x_train, y_train)
        joblib.dump(clf, model_out_path)

        # 5. Log Model with Signature and Input Example
        signature = infer_signature(x_train, clf.predict(x_train))
        input_example = x_train.iloc[:1]
        mlflow.sklearn.log_model(
            clf, "model", signature=signature, input_example=input_example
        )
        mlflow.log_artifact(config.FEATURE_COLUMNS_PATH)
        if os.path.exists(config.SCALER_PATH):
            mlflow.log_artifact(config.SCALER_PATH)
        if os.path.exists(config.ENCODERS_PATH):
            mlflow.log_artifact(config.ENCODERS_PATH)
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "model_type": "RandomForest",
                "smote": True,
                "test_size": test_size,
            }
        )

        # 6. Evaluate
        y_pred = clf.predict(x_test)
        y_proba = clf.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        logger.info("Metrics: %s", metrics)

        # 7. Save test set for evaluation
        x_test.to_csv(config.X_TEST_PATH, index=False)
        y_test.to_csv(config.Y_TEST_PATH, index=False)

        with open(config.RUN_ID_PATH, "w") as f:
            f.write(run.info.run_id)
    return clf


if __name__ == "__main__":
    train_model()
