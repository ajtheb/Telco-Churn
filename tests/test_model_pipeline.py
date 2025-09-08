import sys
sys.path.append('./src')
import pandas as pd
import joblib
from model_training import train_model

def test_training_pipeline():
    clf = train_model()
    assert hasattr(clf, "predict"), "Model does not have predict method"

def test_model_output_shape():
    X_test = pd.read_csv('data/processed/X_test.csv')
    clf = joblib.load('models/churn_rf.pkl')
    preds = clf.predict(X_test)
    assert len(preds) == len(X_test), "Prediction shape mismatch"

if __name__ == "__main__":
    test_training_pipeline()
    test_model_output_shape()
    print("All tests passed.")
