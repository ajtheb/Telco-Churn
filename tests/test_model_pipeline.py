import sys
import pandas as pd
import joblib
sys.path.append("./src")
from model_training import train_model



def test_training_pipeline():
    clf = train_model()
    assert hasattr(clf, "predict"), "Model does not have predict method"


def test_model_output_shape():
    x_test = pd.read_csv("data/processed/X_test.csv")
    clf = joblib.load("models/churn_rf.pkl")
    preds = clf.predict(x_test)
    assert len(preds) == len(x_test), "Prediction shape mismatch"


if __name__ == "__main__":
    test_training_pipeline()
    test_model_output_shape()
    print("All tests passed.")
