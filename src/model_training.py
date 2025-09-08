import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# Ensure directories exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./data/processed", exist_ok=True)


def train_model(
    data_path="./data/processed/featured_telco.csv",
    model_out_path="./models/churn_rf.pkl",
    mlflow_experiment="telco-churn",
    n_estimators=200,
    test_size=0.2,
    random_state=42,
):
    # Debug: Print working directory and URIs
    print("Current working directory:", os.getcwd())
    print("Tracking URI:", mlflow.get_tracking_uri())

    # 1. Load Data
    df = pd.read_csv(data_path)
    x = df.drop(["Churn"], axis=1)
    y = df["Churn"]

    # 2. Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # 3. Define Model
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # 4. Train Model with MLflow Tracking
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        clf.fit(x_train, y_train)
        joblib.dump(clf, model_out_path)

        # 5. Log Model with Signature and Input Example
        signature = infer_signature(x_train, clf.predict(x_train))
        input_example = x_train.iloc[:1]
        mlflow.sklearn.log_model(
            clf, "model", signature=signature, input_example=input_example
        )
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
        print("Metrics:", metrics)

        # 7. Save test set for evaluation
        x_test.to_csv("./data/processed/X_test.csv", index=False)
        y_test.to_csv("./data/processed/y_test.csv", index=False)

    return clf


if __name__ == "__main__":
    train_model()
